#include <contact_system/vertex_half_plane_frictional_contact.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <utils/make_spd.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
namespace
{
template <typename ContactSink>
void record_friction_ph_contact_topology(cudaStream_t               stream,
                                         ContactSink                structured_sink,
                                         muda::CBufferView<Vector2i> PHs)
{
    if(PHs.size() == 0)
        return;

    using namespace muda;
    using Store = VertexHalfPlaneFrictionalContact::StoreScalar;
    ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(PHs.size(),
               [structured_sink,
                PHs = PHs.viewer().name("friction_PHs")] __device__(
                   int I) mutable
               {
                   const Vector2i PH = PHs(I);
                   structured_sink.write_weighted_hessian(PH(0), Store{1});
               });
}

template <typename ContactSink>
void record_friction_ph_contact_weights(
    cudaStream_t                       stream,
    ContactSink                        structured_sink,
    muda::CBufferView<Vector2i>         PHs,
    muda::CBuffer2DView<ContactCoeff>   table,
    muda::CBufferView<IndexT>           contact_ids,
    IndexT                              half_plane_vertex_offset,
    Float                               dt)
{
    if(PHs.size() == 0)
        return;

    using namespace muda;
    using Store = VertexHalfPlaneFrictionalContact::StoreScalar;
    ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(PHs.size(),
               [structured_sink,
                PHs = PHs.viewer().name("friction_PHs"),
                table = table.viewer().name("contact_tabular"),
                contact_ids = contact_ids.viewer().name("contact_element_ids"),
                half_plane_vertex_offset,
                dt] __device__(int I) mutable
               {
                   const Vector2i PH = PHs(I);
                   const IndexT   vI = PH(0);
                   const IndexT   HI = PH(1);
                   const ContactCoeff coeff =
                       table(contact_ids(vI), contact_ids(HI + half_plane_vertex_offset));
                   const Store weight =
                       safe_cast<Store>(coeff.kappa * coeff.mu * dt * dt);
                   structured_sink.write_weighted_hessian(vI, weight);
               });
}
}  // namespace

void VertexHalfPlaneFrictionalContact::do_build(ContactReporter::BuildInfo& info)
{
    auto& config      = world().scene().config();
    auto  enable_attr = config.find<IndexT>("contact/friction/enable");
    auto  dt_attr     = config.find<Float>("dt");

    if(!enable_attr->view()[0])
    {
        throw SimSystemException("Frictional contact is disabled");
    }

    m_impl.global_trajectory_filter = require<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager   = require<GlobalContactManager>();
    m_impl.global_vertex_manager    = require<GlobalVertexManager>();
    m_impl.vertex_reporter          = require<HalfPlaneVertexReporter>();
    m_impl.dt                       = dt_attr->view()[0];

    BuildInfo this_info;
    do_build(this_info);

    on_init_scene(
        [this]
        {
            m_impl.veretx_half_plane_trajectory_filter =
                m_impl.global_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();
        });
}

void VertexHalfPlaneFrictionalContact::do_report_gradient_hessian_extent(
    GlobalContactManager::GradientHessianExtentInfo& info)
{
    auto& filter = m_impl.veretx_half_plane_trajectory_filter;

    SizeT count = filter->friction_PHs().size();

    info.gradient_count(count);

    if(info.gradient_only())
        return;

    info.hessian_count(count);
}

void VertexHalfPlaneFrictionalContact::do_report_energy_extent(GlobalContactManager::EnergyExtentInfo& info)
{
    auto& filter = m_impl.veretx_half_plane_trajectory_filter;

    SizeT count     = filter->friction_PHs().size();
    m_impl.PH_count = count;

    info.energy_count(count);
}

void VertexHalfPlaneFrictionalContact::do_compute_energy(GlobalContactManager::EnergyInfo& info)
{
    using namespace muda;

    EnergyInfo this_info{&m_impl};
    this_info.m_energies = info.energies();
    m_impl.energies      = this_info.m_energies;

    do_compute_energy(this_info);
}

void VertexHalfPlaneFrictionalContact::do_assemble(GlobalContactManager::GradientHessianInfo& info)
{
    ContactInfo this_info{&m_impl};
    this_info.m_gradient_only = info.gradient_only();

    this_info.m_gradients = info.gradients();
    m_impl.gradients      = this_info.m_gradients;
    this_info.m_hessians  = info.hessians();
    m_impl.hessians       = this_info.m_hessians;

    // let subclass to fill in the data
    do_assemble(this_info);
}

bool VertexHalfPlaneFrictionalContact::do_supports_structured_hessian() const
{
    return true;
}

void VertexHalfPlaneFrictionalContact::do_assemble_structured_hessian(
    GlobalDyTopoEffectManager::StructuredHessianInfo& info)
{
    ContactInfo this_info{&m_impl};
    this_info.m_gradient_only      = false;
    this_info.m_hessian_only       = true;
    this_info.m_structured_hessian = true;
    this_info.m_structured_sink    = info.contact_sink();

    if(this_info.m_structured_sink.topology_probe_only())
    {
        record_friction_ph_contact_topology(info.stream(),
                                            this_info.m_structured_sink,
                                            this_info.friction_PHs());
        return;
    }

    if(this_info.m_structured_sink.approximate_weight_probe_only())
    {
        record_friction_ph_contact_weights(info.stream(),
                                           this_info.m_structured_sink,
                                           this_info.friction_PHs(),
                                           this_info.contact_tabular(),
                                           this_info.contact_element_ids(),
                                           this_info.half_plane_vertex_offset(),
                                           this_info.dt());
        return;
    }

    const auto matrix = this_info.m_structured_sink.sink.matrix;
    if(matrix.native_enabled())
    {
        const auto descriptors = info.vertex_descriptors();
        if(descriptors.data() != nullptr && matrix.old_to_chain.data() != nullptr
           && matrix.horizon != 0 && matrix.block_size != 0)
        {
            m_impl.loose_resize(m_impl.PH_native_contact_targets,
                                this_info.friction_PHs().size() * PHHalfHessianSize);

            rebuild_socu_native_vertex_half_plane_contact_targets(
                info.stream(),
                m_impl.PH_native_contact_targets.view(),
                this_info.friction_PHs(),
                descriptors,
                matrix.old_to_chain,
                this_info.m_structured_sink.abd_vertex_to_J,
                matrix.horizon,
                matrix.block_size,
                this_info.m_structured_sink.offband_policy);

            this_info.m_PH_native_contact_targets =
                m_impl.PH_native_contact_targets.view().as_const();
        }
    }

    m_impl.hessians = {};
    do_assemble(this_info);
}

muda::CBuffer2DView<ContactCoeff> VertexHalfPlaneFrictionalContact::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

muda::CBufferView<Vector2i> VertexHalfPlaneFrictionalContact::BaseInfo::friction_PHs() const
{
    return m_impl->veretx_half_plane_trajectory_filter->friction_PHs();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContact::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

muda::CBufferView<Float> VertexHalfPlaneFrictionalContact::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContact::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

muda::CBufferView<Vector3> VertexHalfPlaneFrictionalContact::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

muda::CBufferView<IndexT> VertexHalfPlaneFrictionalContact::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

muda::CBufferView<IndexT> VertexHalfPlaneFrictionalContact::BaseInfo::subscene_element_ids() const
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

Float VertexHalfPlaneFrictionalContact::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

muda::CBufferView<Float> VertexHalfPlaneFrictionalContact::BaseInfo::d_hats() const
{
    return m_impl->global_vertex_manager->d_hats();
}

Float VertexHalfPlaneFrictionalContact::BaseInfo::dt() const
{
    return m_impl->dt;
}

Float VertexHalfPlaneFrictionalContact::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}

IndexT VertexHalfPlaneFrictionalContact::BaseInfo::half_plane_vertex_offset() const
{
    return m_impl->vertex_reporter->vertex_offset();
}

muda::BufferView<VertexHalfPlaneFrictionalContact::EnergyScalar> VertexHalfPlaneFrictionalContact::EnergyInfo::energies() const noexcept
{
    return m_energies;
}

muda::CBufferView<Vector2i> VertexHalfPlaneFrictionalContact::PHs() const noexcept
{
    return m_impl.veretx_half_plane_trajectory_filter->friction_PHs();
}

muda::CBufferView<VertexHalfPlaneFrictionalContact::EnergyScalar> VertexHalfPlaneFrictionalContact::energies() const noexcept
{
    return m_impl.energies;
}

muda::CDoubletVectorView<VertexHalfPlaneFrictionalContact::StoreScalar, 3> VertexHalfPlaneFrictionalContact::gradients() const noexcept
{
    return m_impl.gradients;
}

muda::CTripletMatrixView<VertexHalfPlaneFrictionalContact::StoreScalar, 3> VertexHalfPlaneFrictionalContact::hessians() const noexcept
{
    return m_impl.hessians;
}
}  // namespace uipc::backend::cuda_mixed

#include <contact_system/contact_exporter.h>

namespace uipc::backend::cuda_mixed
{
using StoreVec3 = Eigen::Matrix<VertexHalfPlaneFrictionalContact::StoreScalar, 3, 1>;
using StoreMat3 = Eigen::Matrix<VertexHalfPlaneFrictionalContact::StoreScalar, 3, 3>;
class VertexHalfPlaneFrictionalContactExporter : public ContactExporter
{
  public:
    using ContactExporter::ContactExporter;

    SimSystemSlot<VertexHalfPlaneFrictionalContact> vertex_half_plane_Frictional_contact;
    SimSystemSlot<HalfPlaneVertexReporter> half_plane_vertex_reporter;

    void do_build(BuildInfo& info) override
    {
        vertex_half_plane_Frictional_contact =
            require<VertexHalfPlaneFrictionalContact>(QueryOptions{.exact = false});
        half_plane_vertex_reporter = require<HalfPlaneVertexReporter>();
    }

    std::string_view get_prim_type() const noexcept override { return "PH+F"; }

    void get_contact_energy(std::string_view prim_type, geometry::Geometry& energy_geo) override
    {
        auto PHs      = vertex_half_plane_Frictional_contact->PHs();
        auto energies = vertex_half_plane_Frictional_contact->energies();

        UIPC_ASSERT(PHs.size() == energies.size(), "PHs and energies must have the same size.");

        energy_geo.instances().resize(PHs.size());
        auto topo = energy_geo.instances().find<Vector2i>("topo");
        if(!topo)
        {
            topo = energy_geo.instances().create<Vector2i>("topo", Vector2i::Zero());
        }

        auto topo_view = view(*topo);
        PHs.copy_to(topo_view.data());
        auto v_offset = half_plane_vertex_reporter->vertex_offset();

        for(Vector2i& topo : topo_view)
            topo[1] += v_offset;


        copy_contact_energies_to_geometry(energies, energy_geo);
    }

    void get_contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad) override
    {
        auto PH_grads = vertex_half_plane_Frictional_contact->gradients();
        vert_grad.instances().resize(PH_grads.doublet_count());
        auto i = vert_grad.instances().find<IndexT>("i");
        if(!i)
        {
            i = vert_grad.instances().create<IndexT>("i", -1);
        }
        auto i_view = view(*i);
        PH_grads.indices().copy_to(i_view.data());

        auto grad = vert_grad.instances().find<StoreVec3>("grad");
        if(!grad)
        {
            grad = vert_grad.instances().create<StoreVec3>("grad", StoreVec3::Zero());
        }
        auto grad_view = view(*grad);
        PH_grads.values().copy_to(grad_view.data());
    }

    void get_contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess) override
    {
        auto PH_hess = vertex_half_plane_Frictional_contact->hessians();
        vert_hess.instances().resize(PH_hess.triplet_count());
        auto i = vert_hess.instances().find<IndexT>("i");
        if(!i)
        {
            i = vert_hess.instances().create<IndexT>("i", -1);
        }
        auto i_view = view(*i);
        PH_hess.row_indices().copy_to(i_view.data());

        auto j = vert_hess.instances().find<IndexT>("j");
        if(!j)
        {
            j = vert_hess.instances().create<IndexT>("j", -1);
        }
        auto j_view = view(*j);
        PH_hess.col_indices().copy_to(j_view.data());

        auto hess = vert_hess.instances().find<StoreMat3>("hess");
        if(!hess)
        {
            hess = vert_hess.instances().create<StoreMat3>("hess", StoreMat3::Zero());
        }
        auto hess_view = view(*hess);
        PH_hess.values().copy_to(hess_view.data());
    }
};

REGISTER_SIM_SYSTEM(VertexHalfPlaneFrictionalContactExporter);
}  // namespace uipc::backend::cuda_mixed
