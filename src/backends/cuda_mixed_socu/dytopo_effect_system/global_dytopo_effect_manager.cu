#include <sim_engine.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <contact_system/contact_reporter.h>
#include <contact_system/global_contact_manager.h>
#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/simplex_normal_contact.h>
#include <contact_system/vertex_half_plane_frictional_contact.h>
#include <contact_system/vertex_half_plane_normal_contact.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <inter_primitive_effect_system/inter_primitive_constitution_manager.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <linear_system/socu_contact_assembly_plan.h>
#include <linear_system/socu_contact_direct_evaluator.h>
#include <linear_system/socu_contact_executor.h>
#include <linear_system/socu_contact_topology_stamp.h>
#include <uipc/common/timer.h>
#include <uipc/common/enumerate.h>
#include <kernel_cout.h>
#include <uipc/common/unit.h>
#include <uipc/common/zip.h>
#include <energy_component_flags.h>
#include <fmt/format.h>
#include <muda/buffer/buffer_launch.h>
#include <chrono>
#include <string_view>
#include <type_traits>
#include <vector>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda_mixed::GlobalDyTopoEffectManager>
{
  public:
    static U<cuda_mixed::GlobalDyTopoEffectManager> create(cuda_mixed::SimEngine& engine)
    {
        auto dytopo_effect_enable_attr =
            engine.world().scene().config().find<IndexT>("contact/enable");
        bool dytopo_effect_enable = dytopo_effect_enable_attr->view()[0] != 0;

        auto& types = engine.world().scene().constitution_tabular().types();
        bool  has_inter_primitive_constitution =
            types.find(std::string{builtin::InterPrimitive}) != types.end();

        if(dytopo_effect_enable || has_inter_primitive_constitution)
            return make_unique<cuda_mixed::GlobalDyTopoEffectManager>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda_mixed
{
namespace
{
const char* dytopo_assemble_timer_name(const DyTopoEffectReporter& reporter)
{
    if(dynamic_cast<const ContactReporter*>(&reporter))
        return "Assemble Contact";
    if(dynamic_cast<const InterPrimitiveConstitutionManager*>(&reporter))
        return "Assemble Inter-Primitive";
    return "Assemble Unclassified DyTopo";
}

constexpr SizeT ContactSignatureFnvOffset =
    static_cast<SizeT>(1469598103934665603ull);
constexpr SizeT ContactSignatureFnvPrime =
    static_cast<SizeT>(1099511628211ull);

void mix_contact_signature(SizeT& signature, SizeT value) noexcept
{
    signature ^= value;
    signature *= ContactSignatureFnvPrime;
}

template <typename ValueT>
void mix_contact_vector_view(SizeT& signature,
                             SizeT  tag,
                             muda::CBufferView<ValueT> view)
{
    mix_contact_signature(signature, tag);
    mix_contact_signature(signature, view.size());
    if(view.size() == 0)
        return;

    std::vector<ValueT> host(view.size());
    view.copy_to(host.data());
    for(const auto& item : host)
    {
        for(Eigen::Index i = 0; i < item.size(); ++i)
            mix_contact_signature(signature, static_cast<SizeT>(item(i)));
    }
}

void check_native_contact_cuda(cudaError_t error, std::string_view operation)
{
    if(error != cudaSuccess)
    {
        throw SimSystemException{fmt::format(
            "{} failed during SOCU native contact executor replay: {}",
            operation,
            cudaGetErrorString(error))};
    }
}

struct NativeContactTimingEvents
{
    cudaEvent_t hessian_start = nullptr;
    cudaEvent_t hessian_done = nullptr;
    cudaEvent_t executor_start = nullptr;
    cudaEvent_t executor_done = nullptr;
    cudaEvent_t hot_reduce_start = nullptr;
    cudaEvent_t hot_reduce_done = nullptr;
    cudaEvent_t compare_start = nullptr;
    cudaEvent_t compare_done = nullptr;

    ~NativeContactTimingEvents() noexcept { destroy(); }

    void create()
    {
        check_native_contact_cuda(cudaEventCreate(&hessian_start),
                                  "cudaEventCreate(hessian_start)");
        check_native_contact_cuda(cudaEventCreate(&hessian_done),
                                  "cudaEventCreate(hessian_done)");
        check_native_contact_cuda(cudaEventCreate(&executor_start),
                                  "cudaEventCreate(executor_start)");
        check_native_contact_cuda(cudaEventCreate(&executor_done),
                                  "cudaEventCreate(executor_done)");
        check_native_contact_cuda(cudaEventCreate(&hot_reduce_start),
                                  "cudaEventCreate(hot_reduce_start)");
        check_native_contact_cuda(cudaEventCreate(&hot_reduce_done),
                                  "cudaEventCreate(hot_reduce_done)");
        check_native_contact_cuda(cudaEventCreate(&compare_start),
                                  "cudaEventCreate(compare_start)");
        check_native_contact_cuda(cudaEventCreate(&compare_done),
                                  "cudaEventCreate(compare_done)");
    }

    void destroy() noexcept
    {
        if(hessian_start)
            cudaEventDestroy(hessian_start);
        if(hessian_done)
            cudaEventDestroy(hessian_done);
        if(executor_start)
            cudaEventDestroy(executor_start);
        if(executor_done)
            cudaEventDestroy(executor_done);
        if(hot_reduce_start)
            cudaEventDestroy(hot_reduce_start);
        if(hot_reduce_done)
            cudaEventDestroy(hot_reduce_done);
        if(compare_start)
            cudaEventDestroy(compare_start);
        if(compare_done)
            cudaEventDestroy(compare_done);
        hessian_start = nullptr;
        hessian_done = nullptr;
        executor_start = nullptr;
        executor_done = nullptr;
        hot_reduce_start = nullptr;
        hot_reduce_done = nullptr;
        compare_start = nullptr;
        compare_done = nullptr;
    }
};

}  // namespace

REGISTER_SIM_SYSTEM(GlobalDyTopoEffectManager);

muda::CBCOOVectorView<GlobalDyTopoEffectManager::StoreScalar, 3>
GlobalDyTopoEffectManager::gradients() const noexcept
{
    return m_impl.sorted_dytopo_effect_gradient.view();
}

muda::CBCOOMatrixView<GlobalDyTopoEffectManager::StoreScalar, 3>
GlobalDyTopoEffectManager::hessians() const noexcept
{
    return m_impl.sorted_dytopo_effect_hessian.view();
}

void GlobalDyTopoEffectManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager = require<GlobalVertexManager>();
    m_impl.abd_linear_subsystem = find<ABDLinearSubsystem>();
    m_impl.fem_linear_subsystem = find<FEMLinearSubsystem>();
    m_impl.affine_body_dynamics = find<AffineBodyDynamics>();
    m_impl.finite_element_method = find<FiniteElementMethod>();
    m_impl.affine_body_vertex_reporter = find<AffineBodyVertexReporter>();
    m_impl.finite_element_vertex_reporter = find<FiniteElementVertexReporter>();
    m_impl.global_contact_manager = find<GlobalContactManager>();
    m_impl.half_plane = find<HalfPlane>();
    m_impl.half_plane_vertex_reporter = find<HalfPlaneVertexReporter>();
    if(auto dt_attr = config.find<Float>("dt"))
        m_impl.dt = dt_attr->view()[0];
}

void GlobalDyTopoEffectManager::Impl::init(WorldVisitor& world)
{
    // 3) reporters
    auto dytopo_effect_reporter_view = dytopo_effect_reporters.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->m_index = i;

    reporter_energy_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_gradient_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_hessian_offsets_counts.resize(dytopo_effect_reporter_view.size());

    // 4) receivers
    auto dytopo_effect_receiver_view = dytopo_effect_receivers.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->m_index = i;

    classified_dytopo_effect_gradients.resize(dytopo_effect_receiver_view.size());
    classified_dytopo_effect_hessians.resize(dytopo_effect_receiver_view.size());
}

void GlobalDyTopoEffectManager::Impl::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    _assemble(info);
    _convert_matrix(info);
    _distribute(info);
}

void GlobalDyTopoEffectManager::Impl::_assemble(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Assemble Dytopo Effect"};

    auto vertex_count = global_vertex_manager->positions().size();

    auto reporter_gradient_counts = reporter_gradient_offsets_counts.counts();
    auto reporter_hessian_counts  = reporter_hessian_offsets_counts.counts();
    const bool structured_hessian_direct =
        info.m_assembly_mode == NewtonAssemblyMode::GradientStructuredHessian;
    bool gradient_only = info.m_gradient_only || structured_hessian_direct;

    logger::info("DyTopo Effect Assembly: GradientOnly={}, ComponentFlags={}, AssemblyMode={}",
                 info.m_gradient_only,
                 enum_flags_name(info.m_component_flags),
                 newton_assembly_mode_name(info.m_assembly_mode));

    for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
    {
        reporter_gradient_counts[i] = 0;
        reporter_hessian_counts[i]  = 0;

        if(!has_flags(info.m_component_flags, reporter->component_flags()))
            continue;

        if(structured_hessian_direct && !reporter->supports_structured_hessian())
        {
            throw SimSystemException{fmt::format(
                "structured_dytopo_reporter_not_supported: reporter '{}' does not "
                "support direct StructuredAssemblySink Hessian writes",
                reporter->name())};
        }

        GradientHessianExtentInfo extent_info;
        extent_info.m_gradient_only = gradient_only;
        reporter->report_gradient_hessian_extent(extent_info);

        reporter_gradient_counts[i] = extent_info.m_gradient_count;
        reporter_hessian_counts[i] = gradient_only ? 0 : extent_info.m_hessian_count;
        logger::info("<{}> DyTopo Grad3 count: {}, DyTopo Hess3x3 count: {}",
                     reporter->name(),
                     extent_info.m_gradient_count,
                     extent_info.m_hessian_count);
    }

    // scan
    reporter_gradient_offsets_counts.scan();
    reporter_hessian_offsets_counts.scan();

    auto total_gradient_count = reporter_gradient_offsets_counts.total_count();
    auto total_hessian_count  = reporter_hessian_offsets_counts.total_count();

    // allocate
    loose_resize_entries(collected_dytopo_effect_gradient, total_gradient_count);
    loose_resize_entries(sorted_dytopo_effect_gradient, total_gradient_count);
    loose_resize_entries(collected_dytopo_effect_hessian, total_hessian_count);
    loose_resize_entries(sorted_dytopo_effect_hessian, total_hessian_count);
    collected_dytopo_effect_gradient.reshape(vertex_count);
    collected_dytopo_effect_hessian.reshape(vertex_count, vertex_count);

    // collect
    for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
    {
        if(!has_flags(info.m_component_flags, reporter->component_flags()))
            continue;

        auto [g_offset, g_count] = reporter_gradient_offsets_counts[i];
        auto [h_offset, h_count] = reporter_hessian_offsets_counts[i];

        GradientHessianInfo info;
        info.m_gradient_only = gradient_only;

        info.m_gradients =
            collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
        info.m_hessians = collected_dytopo_effect_hessian.view().subview(h_offset, h_count);

        {
            Timer timer{dytopo_assemble_timer_name(*reporter)};
            reporter->assemble(info);
        }
    }
}

void GlobalDyTopoEffectManager::Impl::_convert_matrix(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Convert Dytopo Matrix"};

    if(info.m_assembly_mode == NewtonAssemblyMode::GradientStructuredHessian)
    {
        loose_resize_entries(sorted_dytopo_effect_hessian, 0);
        auto vertex_count = global_vertex_manager->positions().size();
        sorted_dytopo_effect_hessian.reshape(vertex_count, vertex_count);
    }
    else
    {
        matrix_converter.convert(collected_dytopo_effect_hessian, sorted_dytopo_effect_hessian);
    }
    matrix_converter.convert(collected_dytopo_effect_gradient, sorted_dytopo_effect_gradient);
}

void GlobalDyTopoEffectManager::Impl::_distribute(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Distribute Dytopo Effect"};

    using namespace muda;

    auto vertex_count = global_vertex_manager->positions().size();
    const bool structured_hessian_direct =
        info.m_assembly_mode == NewtonAssemblyMode::GradientStructuredHessian;

    for(auto&& [i, receiver] : enumerate(dytopo_effect_receivers.view()))
    {
        DyTopoClassifyInfo classify_info;
        receiver->report(classify_info);


        ClassifiedDyTopoEffectInfo classified_info;
        auto& classified_gradients = classified_dytopo_effect_gradients[i];
        classified_gradients.reshape(vertex_count);
        auto& classified_hessians = classified_dytopo_effect_hessians[i];
        classified_hessians.reshape(vertex_count, vertex_count);

        // 1) report gradient
        if(classify_info.is_diag())
        {
            const auto N = sorted_dytopo_effect_gradient.doublet_count();

            // clear the range in device
            gradient_range = Vector2i{0, 0};

            // partition
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [gradient_range = gradient_range.viewer().name("gradient_range"),
                     dytopo_effect_gradient =
                         std::as_const(sorted_dytopo_effect_gradient).viewer().name("dytopo_effect_gradient"),
                     range = classify_info.gradient_i_range()] __device__(int I) mutable
                    {
                        auto in_range = [](int i, const Vector2i& range)
                        { return i >= range.x() && i < range.y(); };

                        auto&& [i, G]      = dytopo_effect_gradient(I);
                        bool this_in_range = in_range(i, range);

                        if(!this_in_range)
                        {
                            return;
                        }

                        bool prev_in_range = false;
                        if(I > 0)
                        {
                            auto&& [prev_i, prev_G] = dytopo_effect_gradient(I - 1);
                            prev_in_range = in_range(prev_i, range);
                        }
                        bool next_in_range = false;
                        if(I < dytopo_effect_gradient.total_doublet_count() - 1)
                        {
                            auto&& [next_i, next_G] = dytopo_effect_gradient(I + 1);
                            next_in_range = in_range(next_i, range);
                        }

                        // if the prev is not in range, then this is the start of the partition
                        if(!prev_in_range)
                        {
                            gradient_range->x() = I;
                        }
                        // if the next is not in range, then this is the end of the partition
                        if(!next_in_range)
                        {
                            gradient_range->y() = I + 1;
                        }
                    });

            Vector2i h_range = gradient_range;  // copy back

            auto count = h_range.y() - h_range.x();

            loose_resize_entries(classified_gradients, count);

            // fill
            if(count > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(count,
                           [dytopo_effect_gradient = std::as_const(sorted_dytopo_effect_gradient)
                                                         .viewer()
                                                         .name("dytopo_effect_gradient"),
                            classified_gradient = classified_gradients.viewer().name("classified_gradient"),
                            range = h_range] __device__(int I) mutable
                           {
                               auto&& [i, G] = dytopo_effect_gradient(range.x() + I);
                               classified_gradient(I).write(i, G);
                           });
            }

            classified_info.m_gradients = classified_gradients.view();
        }

        // 2) report hessian
        if(!structured_hessian_direct && !info.m_gradient_only
           && !classify_info.is_empty())
        {
            if(info.m_assembly_mode == NewtonAssemblyMode::GradientStructuredHessian)
            {
                const auto N = collected_dytopo_effect_hessian.triplet_count();

                // +1 for calculate the total count
                loose_resize(selected_hessian, N + 1);
                loose_resize(selected_hessian_offsets, N + 1);

                // select
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(
                        N,
                        [selected_hessian =
                             selected_hessian.view(0, N).viewer().name("selected_hessian"),
                         last =
                             VarView<IndexT>{selected_hessian.data() + N}.viewer().name("last"),
                         dytopo_effect_hessian = collected_dytopo_effect_hessian.cviewer().name(
                             "dytopo_effect_hessian"),
                         i_range = classify_info.hessian_i_range(),
                         j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                        {
                            auto&& [i, j, H] = dytopo_effect_hessian(I);

                            auto in_range = [](int i, const Vector2i& range)
                            { return i >= range.x() && i < range.y(); };

                            selected_hessian(I) =
                                in_range(i, i_range) && in_range(j, j_range) ? 1 : 0;

                            // fill the last one as 0, so that we can calculate the total count
                            // during the exclusive scan
                            if(I == 0)
                                last = 0;
                        });

                // scan
                DeviceScan().ExclusiveSum(selected_hessian.data(),
                                          selected_hessian_offsets.data(),
                                          selected_hessian.size());

                IndexT h_total_count = 0;
                VarView<IndexT>{selected_hessian_offsets.data() + N}.copy_to(&h_total_count);

                loose_resize_entries(classified_hessians, h_total_count);

                // fill
                if(h_total_count > 0)
                {
                    ParallelFor()
                        .file_line(__FILE__, __LINE__)
                        .apply(N,
                               [selected_hessian =
                                    selected_hessian.cviewer().name("selected_hessian"),
                                selected_hessian_offsets =
                                    selected_hessian_offsets.cviewer().name("selected_hessian_offsets"),
                                dytopo_effect_hessian =
                                    collected_dytopo_effect_hessian.cviewer().name(
                                        "dytopo_effect_hessian"),
                                classified_hessian =
                                    classified_hessians.viewer().name("classified_hessian"),
                                i_range = classify_info.hessian_i_range(),
                                j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                               {
                                   if(selected_hessian(I))
                                   {
                                       auto&& [i, j, H] = dytopo_effect_hessian(I);
                                       auto offset = selected_hessian_offsets(I);

                                       classified_hessian(offset).write(i, j, H);
                                   }
                               });
                }

                classified_info.m_hessians = classified_hessians.view();
            }
            else
            {
                const auto N = sorted_dytopo_effect_hessian.triplet_count();

                // +1 for calculate the total count
                loose_resize(selected_hessian, N + 1);
                loose_resize(selected_hessian_offsets, N + 1);

                // select
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(
                        N,
                        [selected_hessian =
                             selected_hessian.view(0, N).viewer().name("selected_hessian"),
                         last =
                             VarView<IndexT>{selected_hessian.data() + N}.viewer().name("last"),
                         dytopo_effect_hessian =
                             sorted_dytopo_effect_hessian.cviewer().name("dytopo_effect_hessian"),
                         i_range = classify_info.hessian_i_range(),
                         j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                        {
                            auto&& [i, j, H] = dytopo_effect_hessian(I);

                            auto in_range = [](int i, const Vector2i& range)
                            { return i >= range.x() && i < range.y(); };

                            selected_hessian(I) =
                                in_range(i, i_range) && in_range(j, j_range) ? 1 : 0;

                            // fill the last one as 0, so that we can calculate the total count
                            // during the exclusive scan
                            if(I == 0)
                                last = 0;
                        });

                // scan
                DeviceScan().ExclusiveSum(selected_hessian.data(),
                                          selected_hessian_offsets.data(),
                                          selected_hessian.size());

                IndexT h_total_count = 0;
                VarView<IndexT>{selected_hessian_offsets.data() + N}.copy_to(&h_total_count);

                loose_resize_entries(classified_hessians, h_total_count);

                // fill
                if(h_total_count > 0)
                {
                    ParallelFor()
                        .file_line(__FILE__, __LINE__)
                        .apply(N,
                               [selected_hessian =
                                    selected_hessian.cviewer().name("selected_hessian"),
                                selected_hessian_offsets =
                                    selected_hessian_offsets.cviewer().name("selected_hessian_offsets"),
                                dytopo_effect_hessian =
                                    sorted_dytopo_effect_hessian.cviewer().name(
                                        "dytopo_effect_hessian"),
                                classified_hessian =
                                    classified_hessians.viewer().name("classified_hessian"),
                                i_range = classify_info.hessian_i_range(),
                                j_range = classify_info.hessian_j_range()] __device__(int I) mutable
                               {
                                   if(selected_hessian(I))
                                   {
                                       auto&& [i, j, H] = dytopo_effect_hessian(I);
                                       auto offset = selected_hessian_offsets(I);

                                       classified_hessian(offset).write(i, j, H);
                                   }
                               });
                }

                classified_info.m_hessians = classified_hessians.view();
            }
        }

        receiver->receive(classified_info);
    }
}

void GlobalDyTopoEffectManager::Impl::assemble_structured_hessian(
    GlobalLinearSystem::StructuredAssemblyInfo& structured_info)
{
    using namespace muda;

    if(dytopo_effect_reporters.view().empty())
        return;

    ensure_structured_vertex_descriptors(structured_info);

    StructuredHessianInfo info;
    info.m_stream = structured_info.stream();
    auto contact_sink = structured_info.sink();
    info.m_contact_sink.sink = contact_sink;
    info.m_contact_sink.counters = structured_info.contact_counters();
    info.m_contact_sink.hessian_cache = structured_info.contact_hessian_cache();
    info.m_contact_sink.offband_policy = structured_info.contact_offband_policy();
    if(structured_vertex_descriptor_epoch == structured_info.descriptor_epoch())
    {
        info.m_vertex_descriptors =
            structured_vertex_descriptors.view().as_const();
        structured_info.set_native_vertex_descriptors(info.m_vertex_descriptors);
        info.m_descriptor_epoch = structured_vertex_descriptor_epoch;
    }

    if(abd_linear_subsystem && affine_body_dynamics && affine_body_vertex_reporter)
    {
        info.m_contact_sink.abd_vertex_offset =
            affine_body_vertex_reporter->vertex_offset();
        info.m_contact_sink.abd_vertex_count =
            affine_body_vertex_reporter->vertex_count();
        auto abd_body_is_fixed = affine_body_dynamics->body_is_fixed();
        info.m_contact_sink.abd_body_count =
            static_cast<IndexT>(abd_body_is_fixed.size());
        info.m_contact_sink.abd_old_dof_offset =
            abd_linear_subsystem->dof_offset();
        info.m_contact_sink.abd_vertex_to_body =
            affine_body_dynamics->v2b();
        info.m_contact_sink.abd_vertex_to_J =
            affine_body_dynamics->Js();
        info.m_contact_sink.abd_body_is_fixed = abd_body_is_fixed;
    }

    if(fem_linear_subsystem && finite_element_method && finite_element_vertex_reporter)
    {
        info.m_contact_sink.fem_vertex_offset =
            finite_element_vertex_reporter->vertex_offset();
        info.m_contact_sink.fem_vertex_count =
            finite_element_vertex_reporter->vertex_count();
        info.m_contact_sink.fem_old_dof_offset =
            fem_linear_subsystem->dof_offset();
        info.m_contact_sink.fem_vertex_is_fixed =
            finite_element_method->is_fixed();
    }

    if(info.m_contact_sink.hessian_cache.replay_valid())
    {
        structured_info.set_native_contact_replay_path("legacy_structured");
        Timer timer{"Replay Structured Contact Hessian Cache"};
        replay_structured_contact_hessian_cache(structured_info.stream(),
                                                info.m_contact_sink);
        return;
    }

    auto assemble_non_contact_structured_reporters = [&]()
    {
        for(auto&& reporter : dytopo_effect_reporters.view())
        {
            if(has_flags(EnergyComponentFlags::Contact,
                         reporter->component_flags()))
                continue;
            if(!reporter->supports_structured_hessian())
            {
                throw SimSystemException{fmt::format(
                    "structured_dytopo_reporter_not_supported: reporter '{}' does not "
                    "support direct StructuredAssemblySink Hessian writes",
                    reporter->name())};
            }

            Timer timer{dytopo_assemble_timer_name(*reporter)};
            reporter->assemble_structured_hessian(info);
        }
    };

    if(structured_info.native_contact_plan_executor_enabled())
    {
        auto* plan = structured_info.native_contact_assembly_plan();
        if(!plan)
        {
            throw SimSystemException{
                "socu_native_contact_executor_missing_plan: executor was "
                "enabled without a prepared contact assembly plan"};
        }

        if(socu_contact_assembly_plan_empty(*plan))
        {
            // native_contact_empty_plan_replay: an empty contact topology is a
            // valid native no-op and must not fall back to legacy contact TUs.
            structured_info.set_native_contact_replay_path("native_plan");
            assemble_non_contact_structured_reporters();
            return;
        }

        const auto plan_view = socu_contact_assembly_plan_view(*plan);
        const auto matrix = structured_info.native_matrix();
        if(!plan_view.valid() || !matrix.valid())
        {
            throw SimSystemException{
                "socu_native_contact_executor_invalid_view: executor plan or "
                "native structured matrix view is invalid"};
        }

        NativeContactTimingEvents timing_events;
        timing_events.create();
        const auto begin = std::chrono::steady_clock::now();
        check_native_contact_cuda(
            cudaEventRecord(timing_events.hessian_start,
                            structured_info.stream()),
            "cudaEventRecord(hessian_start)");

        const auto evaluator_path =
            structured_info.native_contact_evaluator_path();

        if(evaluator_path == SocuContactEvaluatorPath::DirectNative
           || evaluator_path == SocuContactEvaluatorPath::Hybrid
           || evaluator_path == SocuContactEvaluatorPath::DirectCompare)
        {
            if(!global_contact_manager)
            {
                throw SimSystemException{
                    "socu_native_contact_direct_missing_contact_manager: "
                    "direct evaluator requires GlobalContactManager"};
            }

            std::vector<SocuContactDirectSourceEntry> direct_sources_host;
            direct_sources_host.reserve(16);
            SocuContactSourceId direct_source_id = 0;
            bool uses_half_plane = false;
            auto push_direct_source = [&](SocuContactModelKind model,
                                          SocuContactFamily family,
                                          auto view)
            {
                SocuContactDirectSourceEntry entry;
                entry.source_id = direct_source_id++;
                entry.model = model;
                entry.family = family;
                entry.contact_count =
                    static_cast<std::uint32_t>(view.size());
                if constexpr(std::is_same_v<std::decay_t<decltype(view)>,
                                            muda::CBufferView<Vector4i>>)
                {
                    entry.stencil_size = 4;
                    entry.stencil4 = view;
                }
                else if constexpr(std::is_same_v<std::decay_t<decltype(view)>,
                                                 muda::CBufferView<Vector3i>>)
                {
                    entry.stencil_size = 3;
                    entry.stencil3 = view;
                }
                else
                {
                    entry.stencil_size = 2;
                    entry.stencil2 = view;
                }
                direct_sources_host.push_back(entry);
            };

            for(auto&& reporter : dytopo_effect_reporters.view())
            {
                if(!has_flags(EnergyComponentFlags::Contact,
                              reporter->component_flags()))
                    continue;

                if(auto* normal = dynamic_cast<SimplexNormalContact*>(reporter))
                {
                    push_direct_source(SocuContactModelKind::SimplexNormal,
                                       SocuContactFamily::PT,
                                       normal->PTs());
                    push_direct_source(SocuContactModelKind::SimplexNormal,
                                       SocuContactFamily::EE,
                                       normal->EEs());
                    push_direct_source(SocuContactModelKind::SimplexNormal,
                                       SocuContactFamily::PE,
                                       normal->PEs());
                    push_direct_source(SocuContactModelKind::SimplexNormal,
                                       SocuContactFamily::PP,
                                       normal->PPs());
                    continue;
                }
                if(auto* friction =
                       dynamic_cast<SimplexFrictionalContact*>(reporter))
                {
                    push_direct_source(SocuContactModelKind::SimplexFrictional,
                                       SocuContactFamily::PT,
                                       friction->PTs());
                    push_direct_source(SocuContactModelKind::SimplexFrictional,
                                       SocuContactFamily::EE,
                                       friction->EEs());
                    push_direct_source(SocuContactModelKind::SimplexFrictional,
                                       SocuContactFamily::PE,
                                       friction->PEs());
                    push_direct_source(SocuContactModelKind::SimplexFrictional,
                                       SocuContactFamily::PP,
                                       friction->PPs());
                    continue;
                }
                if(auto* normal =
                       dynamic_cast<VertexHalfPlaneNormalContact*>(reporter))
                {
                    uses_half_plane = true;
                    push_direct_source(
                        SocuContactModelKind::VertexHalfPlaneNormal,
                        SocuContactFamily::PH,
                        normal->PHs());
                    continue;
                }
                if(auto* friction = dynamic_cast<
                       VertexHalfPlaneFrictionalContact*>(reporter))
                {
                    uses_half_plane = true;
                    push_direct_source(
                        SocuContactModelKind::VertexHalfPlaneFrictional,
                        SocuContactFamily::PH,
                        friction->PHs());
                    continue;
                }

                throw SimSystemException{fmt::format(
                    "socu_native_contact_direct_unsupported_reporter: reporter "
                    "'{}' is a contact reporter but does not expose native "
                    "direct evaluator source views",
                    reporter->name())};
            }

            if(uses_half_plane && (!half_plane || !half_plane_vertex_reporter))
            {
                throw SimSystemException{
                    "socu_native_contact_direct_missing_half_plane_views: "
                    "PH direct evaluator requires HalfPlane and "
                    "HalfPlaneVertexReporter"};
            }

            SocuContactDirectSceneView<StoreScalar> scene;
            scene.contact_tabular =
                global_contact_manager->contact_tabular()
                    .cviewer()
                    .name("socu_direct_contact_tabular");
            scene.positions = global_vertex_manager->positions();
            scene.prev_positions = global_vertex_manager->prev_positions();
            scene.rest_positions = global_vertex_manager->rest_positions();
            scene.thicknesses = global_vertex_manager->thicknesses();
            scene.contact_element_ids =
                global_vertex_manager->contact_element_ids();
            scene.d_hats = global_vertex_manager->d_hats();
            scene.dt = dt;
            scene.eps_velocity = global_contact_manager->eps_velocity();
            if(half_plane)
            {
                scene.half_plane_positions = half_plane->positions();
                scene.half_plane_normals = half_plane->normals();
            }
            if(half_plane_vertex_reporter)
            {
                scene.half_plane_vertex_offset =
                    half_plane_vertex_reporter->vertex_offset();
            }

            muda::DeviceBuffer<SocuContactDirectSourceEntry> direct_sources{
                direct_sources_host};
            SocuContactDirectSourceTable direct_source_table;
            direct_source_table.source_entries =
                direct_sources.view().as_const();
            const SocuContactDirectEvaluator<StoreScalar> direct_evaluator{
                plan_view,
                scene,
                direct_source_table};

            muda::DeviceBuffer<SocuContactEvaluatorSourceEntry<StoreScalar>>
                compare_evaluator_sources;
            SocuContactEvaluatorSourceTable<StoreScalar> compare_sources;
            double compare_triplet_ms = 0.0;
            if(evaluator_path == SocuContactEvaluatorPath::DirectCompare)
            {
                check_native_contact_cuda(
                    cudaEventRecord(timing_events.hessian_start,
                                    structured_info.stream()),
                    "cudaEventRecord(compare_triplet_start)");

                auto vertex_count = global_vertex_manager->positions().size();
                auto reporter_gradient_counts =
                    reporter_gradient_offsets_counts.counts();
                auto reporter_hessian_counts =
                    reporter_hessian_offsets_counts.counts();
                for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
                {
                    reporter_gradient_counts[i] = 0;
                    reporter_hessian_counts[i] = 0;
                    if(!has_flags(EnergyComponentFlags::Contact,
                                  reporter->component_flags()))
                        continue;

                    GradientHessianExtentInfo extent_info;
                    extent_info.m_gradient_only = false;
                    reporter->report_gradient_hessian_extent(extent_info);
                    reporter_gradient_counts[i] = extent_info.m_gradient_count;
                    reporter_hessian_counts[i] = extent_info.m_hessian_count;
                }
                reporter_gradient_offsets_counts.scan();
                reporter_hessian_offsets_counts.scan();

                const auto total_gradient_count =
                    reporter_gradient_offsets_counts.total_count();
                const auto total_hessian_count =
                    reporter_hessian_offsets_counts.total_count();
                loose_resize_entries(collected_dytopo_effect_gradient,
                                     total_gradient_count);
                loose_resize_entries(collected_dytopo_effect_hessian,
                                     total_hessian_count);
                collected_dytopo_effect_gradient.reshape(vertex_count);
                collected_dytopo_effect_hessian.reshape(vertex_count,
                                                       vertex_count);

                for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
                {
                    if(!has_flags(EnergyComponentFlags::Contact,
                                  reporter->component_flags()))
                        continue;

                    const auto [g_offset, g_count] =
                        reporter_gradient_offsets_counts[i];
                    const auto [h_offset, h_count] =
                        reporter_hessian_offsets_counts[i];

                    GradientHessianInfo hessian_info;
                    hessian_info.m_gradient_only = false;
                    hessian_info.m_gradients =
                        collected_dytopo_effect_gradient.view().subview(g_offset,
                                                                        g_count);
                    hessian_info.m_hessians =
                        collected_dytopo_effect_hessian.view().subview(h_offset,
                                                                       h_count);

                    Timer timer{
                        "Assemble Contact Hessian Triplets For SOCU Native Direct Compare"};
                    reporter->assemble(hessian_info);
                }

                check_native_contact_cuda(
                    cudaEventRecord(timing_events.hessian_done,
                                    structured_info.stream()),
                    "cudaEventRecord(compare_triplet_done)");
                check_native_contact_cuda(
                    cudaEventSynchronize(timing_events.hessian_done),
                    "native contact direct compare triplet synchronize");
                float compare_triplet_ms_f = 0.0f;
                check_native_contact_cuda(
                    cudaEventElapsedTime(&compare_triplet_ms_f,
                                         timing_events.hessian_start,
                                         timing_events.hessian_done),
                    "cudaEventElapsedTime(compare_triplet)");
                compare_triplet_ms = static_cast<double>(compare_triplet_ms_f);

                std::vector<SocuContactEvaluatorSourceEntry<StoreScalar>>
                    evaluator_sources_host;
                evaluator_sources_host.reserve(16);
                SocuContactSourceId evaluator_source_id = 0;
                auto push_evaluator_source =
                    [&](SocuContactModelKind model,
                        SocuContactFamily family,
                        muda::CTripletMatrixView<StoreScalar, 3> hessians)
                {
                    SocuContactEvaluatorSourceEntry<StoreScalar> entry;
                    entry.source_id = evaluator_source_id++;
                    entry.model = model;
                    entry.family = family;
                    entry.hessians = hessians;
                    evaluator_sources_host.push_back(entry);
                };

                for(auto&& reporter : dytopo_effect_reporters.view())
                {
                    if(!has_flags(EnergyComponentFlags::Contact,
                                  reporter->component_flags()))
                        continue;

                    if(auto* normal = dynamic_cast<SimplexNormalContact*>(reporter))
                    {
                        push_evaluator_source(SocuContactModelKind::SimplexNormal,
                                              SocuContactFamily::PT,
                                              normal->PT_hessians());
                        push_evaluator_source(SocuContactModelKind::SimplexNormal,
                                              SocuContactFamily::EE,
                                              normal->EE_hessians());
                        push_evaluator_source(SocuContactModelKind::SimplexNormal,
                                              SocuContactFamily::PE,
                                              normal->PE_hessians());
                        push_evaluator_source(SocuContactModelKind::SimplexNormal,
                                              SocuContactFamily::PP,
                                              normal->PP_hessians());
                        continue;
                    }
                    if(auto* friction =
                           dynamic_cast<SimplexFrictionalContact*>(reporter))
                    {
                        push_evaluator_source(
                            SocuContactModelKind::SimplexFrictional,
                            SocuContactFamily::PT,
                            friction->PT_hessians());
                        push_evaluator_source(
                            SocuContactModelKind::SimplexFrictional,
                            SocuContactFamily::EE,
                            friction->EE_hessians());
                        push_evaluator_source(
                            SocuContactModelKind::SimplexFrictional,
                            SocuContactFamily::PE,
                            friction->PE_hessians());
                        push_evaluator_source(
                            SocuContactModelKind::SimplexFrictional,
                            SocuContactFamily::PP,
                            friction->PP_hessians());
                        continue;
                    }
                    if(auto* normal =
                           dynamic_cast<VertexHalfPlaneNormalContact*>(reporter))
                    {
                        push_evaluator_source(
                            SocuContactModelKind::VertexHalfPlaneNormal,
                            SocuContactFamily::PH,
                            normal->hessians());
                        continue;
                    }
                    if(auto* friction = dynamic_cast<
                           VertexHalfPlaneFrictionalContact*>(reporter))
                    {
                        push_evaluator_source(
                            SocuContactModelKind::VertexHalfPlaneFrictional,
                            SocuContactFamily::PH,
                            friction->hessians());
                        continue;
                    }

                    throw SimSystemException{fmt::format(
                        "socu_native_contact_direct_compare_unsupported_reporter: "
                        "reporter '{}' does not expose triplet reference views",
                        reporter->name())};
                }

                compare_evaluator_sources.resize(evaluator_sources_host.size());
                if(!evaluator_sources_host.empty())
                    compare_evaluator_sources.view().copy_from(
                        evaluator_sources_host.data());
                compare_sources.source_entries =
                    compare_evaluator_sources.view().as_const();
            }

            check_native_contact_cuda(
                cudaEventRecord(timing_events.hessian_start,
                                structured_info.stream()),
                "cudaEventRecord(direct_eval_start)");
            muda::DeviceBuffer<SocuDeterministicContactHessian<StoreScalar>>
                direct_hessians;
            direct_hessians.resize(plan_view.programs.size());
            launch_socu_contact_direct_evaluate_programs<StoreScalar>(
                plan_view,
                direct_evaluator,
                direct_hessians.view(),
                structured_info.stream());
            check_native_contact_cuda(cudaGetLastError(),
                                      "native contact direct eval launch");
            check_native_contact_cuda(
                cudaEventRecord(timing_events.hessian_done,
                                structured_info.stream()),
                "cudaEventRecord(direct_eval_done)");

            double direct_compare_ms = 0.0;
            double direct_compare_max_abs_error = 0.0;
            double direct_compare_sum_abs_error = 0.0;
            SizeT  direct_compare_mismatch_count = 0;
            if(evaluator_path == SocuContactEvaluatorPath::DirectCompare)
            {
                muda::DeviceBuffer<SocuContactDirectCompareProgramStats>
                    compare_stats;
                compare_stats.resize(plan_view.programs.size());
                const SocuContactTripletEvaluator<StoreScalar> reference_evaluator{
                    plan_view,
                    compare_sources};
                check_native_contact_cuda(
                    cudaEventRecord(timing_events.compare_start,
                                    structured_info.stream()),
                    "cudaEventRecord(direct_compare_start)");
                launch_socu_contact_compare_direct_triplet_programs<StoreScalar>(
                    plan_view,
                    direct_hessians.view().as_const(),
                    reference_evaluator,
                    compare_stats.view(),
                    Float{1e-4},
                    structured_info.stream());
                check_native_contact_cuda(cudaGetLastError(),
                                          "native contact direct compare launch");
                check_native_contact_cuda(
                    cudaEventRecord(timing_events.compare_done,
                                    structured_info.stream()),
                    "cudaEventRecord(direct_compare_done)");
                check_native_contact_cuda(
                    cudaEventSynchronize(timing_events.compare_done),
                    "native contact direct compare synchronize");

                float direct_compare_ms_f = 0.0f;
                check_native_contact_cuda(
                    cudaEventElapsedTime(&direct_compare_ms_f,
                                         timing_events.compare_start,
                                         timing_events.compare_done),
                    "cudaEventElapsedTime(direct_compare)");
                direct_compare_ms = static_cast<double>(direct_compare_ms_f);

                std::vector<SocuContactDirectCompareProgramStats>
                    compare_stats_host;
                compare_stats.copy_to(compare_stats_host);
                for(const auto& stat : compare_stats_host)
                {
                    direct_compare_max_abs_error =
                        std::max(direct_compare_max_abs_error,
                                 static_cast<double>(stat.max_abs_error));
                    direct_compare_sum_abs_error +=
                        static_cast<double>(stat.sum_abs_error);
                    direct_compare_mismatch_count +=
                        static_cast<SizeT>(stat.mismatch_count);
                }
            }

            check_native_contact_cuda(
                cudaEventRecord(timing_events.executor_start,
                                structured_info.stream()),
                "cudaEventRecord(executor_start)");
            const SocuContactPrecomputedHessianEvaluator<StoreScalar>
                evaluator{direct_hessians.view().as_const()};
            launch_socu_contact_executor_direct_scatter<
                StoreScalar,
                GlobalLinearSystem::SolveScalar>(
                plan_view,
                matrix,
                evaluator,
                {},
                structured_info.stream());
            check_native_contact_cuda(cudaGetLastError(),
                                      "native contact direct scatter launch");
            check_native_contact_cuda(
                cudaEventRecord(timing_events.executor_done,
                                structured_info.stream()),
                "cudaEventRecord(executor_done)");
            check_native_contact_cuda(
                cudaEventRecord(timing_events.hot_reduce_start,
                                structured_info.stream()),
                "cudaEventRecord(hot_reduce_start)");
            launch_socu_contact_executor_hot_reduce<
                StoreScalar,
                GlobalLinearSystem::SolveScalar>(
                plan_view,
                matrix,
                evaluator,
                {},
                structured_info.stream());
            check_native_contact_cuda(cudaGetLastError(),
                                      "native contact hot reduce launch");
            check_native_contact_cuda(
                cudaEventRecord(timing_events.hot_reduce_done,
                                structured_info.stream()),
                "cudaEventRecord(hot_reduce_done)");
            check_native_contact_cuda(
                cudaEventSynchronize(timing_events.hot_reduce_done),
                "native contact executor synchronize");

            float direct_eval_ms = 0.0f;
            float executor_ms = 0.0f;
            float hot_reduce_ms = 0.0f;
            check_native_contact_cuda(
                cudaEventElapsedTime(&direct_eval_ms,
                                     timing_events.hessian_start,
                                     timing_events.hessian_done),
                "cudaEventElapsedTime(direct_eval)");
            check_native_contact_cuda(
                cudaEventElapsedTime(&executor_ms,
                                     timing_events.executor_start,
                                     timing_events.executor_done),
                "cudaEventElapsedTime(executor_scatter)");
            check_native_contact_cuda(
                cudaEventElapsedTime(&hot_reduce_ms,
                                     timing_events.hot_reduce_start,
                                     timing_events.hot_reduce_done),
                "cudaEventElapsedTime(hot_reduce)");
            const auto end = std::chrono::steady_clock::now();
            structured_info.record_native_contact_direct_eval_time_ms(
                static_cast<double>(direct_eval_ms));
            if(evaluator_path == SocuContactEvaluatorPath::DirectCompare)
            {
                structured_info.record_native_contact_hessian_triplet_time_ms(
                    compare_triplet_ms);
                structured_info.record_native_contact_direct_compare_time_ms(
                    direct_compare_ms);
                structured_info.record_native_contact_direct_compare_error(
                    direct_compare_max_abs_error,
                    direct_compare_sum_abs_error,
                    direct_compare_mismatch_count);
            }
            structured_info.record_native_contact_executor_scatter_time_ms(
                static_cast<double>(executor_ms));
            structured_info.record_native_contact_hot_reduce_time_ms(
                static_cast<double>(hot_reduce_ms));
            structured_info.record_native_contact_numeric_time_ms(
                std::chrono::duration<double, std::milli>(end - begin).count());
            structured_info.set_native_contact_replay_path("native_plan");

            assemble_non_contact_structured_reporters();
            return;
        }

        auto vertex_count = global_vertex_manager->positions().size();
        auto reporter_gradient_counts = reporter_gradient_offsets_counts.counts();
        auto reporter_hessian_counts  = reporter_hessian_offsets_counts.counts();
        for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
        {
            reporter_gradient_counts[i] = 0;
            reporter_hessian_counts[i] = 0;
            if(!has_flags(EnergyComponentFlags::Contact,
                          reporter->component_flags()))
                continue;

            GradientHessianExtentInfo extent_info;
            extent_info.m_gradient_only = false;
            reporter->report_gradient_hessian_extent(extent_info);
            reporter_gradient_counts[i] = extent_info.m_gradient_count;
            reporter_hessian_counts[i] = extent_info.m_hessian_count;
        }
        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        const auto total_gradient_count =
            reporter_gradient_offsets_counts.total_count();
        const auto total_hessian_count =
            reporter_hessian_offsets_counts.total_count();
        loose_resize_entries(collected_dytopo_effect_gradient,
                             total_gradient_count);
        loose_resize_entries(collected_dytopo_effect_hessian,
                             total_hessian_count);
        collected_dytopo_effect_gradient.reshape(vertex_count);
        collected_dytopo_effect_hessian.reshape(vertex_count, vertex_count);

        for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
        {
            if(!has_flags(EnergyComponentFlags::Contact,
                          reporter->component_flags()))
                continue;

            const auto [g_offset, g_count] = reporter_gradient_offsets_counts[i];
            const auto [h_offset, h_count] = reporter_hessian_offsets_counts[i];

            GradientHessianInfo hessian_info;
            hessian_info.m_gradient_only = false;
            hessian_info.m_gradients =
                collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
            hessian_info.m_hessians =
                collected_dytopo_effect_hessian.view().subview(h_offset, h_count);

            Timer timer{"Assemble Contact Hessian Triplets For SOCU Native Plan"};
            reporter->assemble(hessian_info);
        }
        check_native_contact_cuda(
            cudaEventRecord(timing_events.hessian_done,
                            structured_info.stream()),
            "cudaEventRecord(hessian_done)");

        std::vector<SocuContactEvaluatorSourceEntry<StoreScalar>>
            evaluator_sources_host;
        evaluator_sources_host.reserve(16);
        SocuContactSourceId evaluator_source_id = 0;
        auto push_evaluator_source = [&](SocuContactModelKind model,
                                         SocuContactFamily family,
                                         muda::CTripletMatrixView<StoreScalar, 3>
                                             hessians)
        {
            SocuContactEvaluatorSourceEntry<StoreScalar> entry;
            entry.source_id = evaluator_source_id++;
            entry.model = model;
            entry.family = family;
            entry.hessians = hessians;
            evaluator_sources_host.push_back(entry);
        };

        for(auto&& reporter : dytopo_effect_reporters.view())
        {
            if(!has_flags(EnergyComponentFlags::Contact,
                          reporter->component_flags()))
                continue;

            if(auto* normal = dynamic_cast<SimplexNormalContact*>(reporter))
            {
                push_evaluator_source(
                    SocuContactModelKind::SimplexNormal,
                    SocuContactFamily::PT,
                    normal->PT_hessians());
                push_evaluator_source(
                    SocuContactModelKind::SimplexNormal,
                    SocuContactFamily::EE,
                    normal->EE_hessians());
                push_evaluator_source(
                    SocuContactModelKind::SimplexNormal,
                    SocuContactFamily::PE,
                    normal->PE_hessians());
                push_evaluator_source(SocuContactModelKind::SimplexNormal,
                                      SocuContactFamily::PP,
                                      normal->PP_hessians());
                continue;
            }
            if(auto* friction = dynamic_cast<SimplexFrictionalContact*>(reporter))
            {
                push_evaluator_source(
                    SocuContactModelKind::SimplexFrictional,
                    SocuContactFamily::PT,
                    friction->PT_hessians());
                push_evaluator_source(
                    SocuContactModelKind::SimplexFrictional,
                    SocuContactFamily::EE,
                    friction->EE_hessians());
                push_evaluator_source(
                    SocuContactModelKind::SimplexFrictional,
                    SocuContactFamily::PE,
                    friction->PE_hessians());
                push_evaluator_source(SocuContactModelKind::SimplexFrictional,
                                      SocuContactFamily::PP,
                                      friction->PP_hessians());
                continue;
            }
            if(auto* normal = dynamic_cast<VertexHalfPlaneNormalContact*>(reporter))
            {
                push_evaluator_source(
                    SocuContactModelKind::VertexHalfPlaneNormal,
                    SocuContactFamily::PH,
                    normal->hessians());
                continue;
            }
            if(auto* friction =
                   dynamic_cast<VertexHalfPlaneFrictionalContact*>(reporter))
            {
                push_evaluator_source(
                    SocuContactModelKind::VertexHalfPlaneFrictional,
                    SocuContactFamily::PH,
                    friction->hessians());
                continue;
            }

            throw SimSystemException{fmt::format(
                "socu_native_contact_executor_unsupported_reporter: reporter "
                "'{}' is a contact reporter but does not expose native "
                "executor Hessian source views",
                reporter->name())};
        }
        muda::DeviceBuffer<SocuContactEvaluatorSourceEntry<StoreScalar>>
            evaluator_sources{evaluator_sources_host};
        SocuContactEvaluatorSourceTable<StoreScalar> sources;
        sources.source_entries = evaluator_sources.view().as_const();

        check_native_contact_cuda(
            cudaEventRecord(timing_events.executor_start,
                            structured_info.stream()),
            "cudaEventRecord(executor_start)");
        const SocuContactTripletEvaluator<StoreScalar> evaluator{plan_view,
                                                                  sources};
        launch_socu_contact_executor_direct_scatter<
            StoreScalar,
            GlobalLinearSystem::SolveScalar>(
            plan_view,
            matrix,
            evaluator,
            {},
            structured_info.stream());
        check_native_contact_cuda(cudaGetLastError(),
                                  "native contact direct scatter launch");
        check_native_contact_cuda(
            cudaEventRecord(timing_events.executor_done,
                            structured_info.stream()),
            "cudaEventRecord(executor_done)");
        check_native_contact_cuda(
            cudaEventRecord(timing_events.hot_reduce_start,
                            structured_info.stream()),
            "cudaEventRecord(hot_reduce_start)");
        launch_socu_contact_executor_hot_reduce<
            StoreScalar,
            GlobalLinearSystem::SolveScalar>(
            plan_view,
            matrix,
            evaluator,
            {},
            structured_info.stream());
        check_native_contact_cuda(cudaGetLastError(),
                                  "native contact hot reduce launch");
        check_native_contact_cuda(
            cudaEventRecord(timing_events.hot_reduce_done,
                            structured_info.stream()),
            "cudaEventRecord(hot_reduce_done)");
        check_native_contact_cuda(
            cudaEventSynchronize(timing_events.hot_reduce_done),
            "native contact executor synchronize");

        float hessian_ms = 0.0f;
        float executor_ms = 0.0f;
        float hot_reduce_ms = 0.0f;
        check_native_contact_cuda(
            cudaEventElapsedTime(&hessian_ms,
                                 timing_events.hessian_start,
                                 timing_events.hessian_done),
            "cudaEventElapsedTime(hessian_triplet)");
        check_native_contact_cuda(
            cudaEventElapsedTime(&executor_ms,
                                 timing_events.executor_start,
                                 timing_events.executor_done),
            "cudaEventElapsedTime(executor_scatter)");
        check_native_contact_cuda(
            cudaEventElapsedTime(&hot_reduce_ms,
                                 timing_events.hot_reduce_start,
                                 timing_events.hot_reduce_done),
            "cudaEventElapsedTime(hot_reduce)");
        const auto end = std::chrono::steady_clock::now();
        structured_info.record_native_contact_hessian_triplet_time_ms(
            static_cast<double>(hessian_ms));
        structured_info.record_native_contact_executor_scatter_time_ms(
            static_cast<double>(executor_ms));
        structured_info.record_native_contact_hot_reduce_time_ms(
            static_cast<double>(hot_reduce_ms));
        structured_info.record_native_contact_numeric_time_ms(
            std::chrono::duration<double, std::milli>(end - begin).count());
        structured_info.set_native_contact_replay_path("native_plan");

        assemble_non_contact_structured_reporters();
        return;
    }

    structured_info.set_native_contact_replay_path("legacy_structured");
    for(auto&& reporter : dytopo_effect_reporters.view())
    {
        if(has_flags(EnergyComponentFlags::Contact, reporter->component_flags()))
        {
            GradientHessianExtentInfo extent_info;
            extent_info.m_gradient_only = false;
            reporter->report_gradient_hessian_extent(extent_info);
            // native_contact_legacy_zero_extent_skip: native-only builds do not
            // compile legacy contact assembly, but zero-contact reporters have
            // no gradient/Hessian work to replay.
            if(extent_info.m_gradient_count == 0 && extent_info.m_hessian_count == 0)
                continue;
        }

        if(!reporter->supports_structured_hessian())
        {
            throw SimSystemException{fmt::format(
                "structured_dytopo_reporter_not_supported: reporter '{}' does not "
                "support direct StructuredAssemblySink Hessian writes",
                reporter->name())};
        }

        Timer timer{dytopo_assemble_timer_name(*reporter)};
        reporter->assemble_structured_hessian(info);
    }
}

void GlobalDyTopoEffectManager::Impl::ensure_structured_vertex_descriptors(
    GlobalLinearSystem::StructuredAssemblyInfo& structured_info)
{
    const IndexT epoch = structured_info.descriptor_epoch();
    if(epoch <= 0)
        return;

    const SizeT global_vertex_count =
        global_vertex_manager ? global_vertex_manager->positions().size() : SizeT{0};
    if(global_vertex_count == 0)
        return;

    IndexT fem_vertex_offset = -1;
    IndexT fem_vertex_count = 0;
    IndexT fem_old_dof_offset = -1;
    muda::CBufferView<IndexT> fem_vertex_is_fixed;
    if(fem_linear_subsystem && finite_element_method && finite_element_vertex_reporter)
    {
        fem_vertex_offset = finite_element_vertex_reporter->vertex_offset();
        fem_vertex_count = finite_element_vertex_reporter->vertex_count();
        fem_old_dof_offset = fem_linear_subsystem->dof_offset();
        fem_vertex_is_fixed = finite_element_method->is_fixed();
    }

    IndexT abd_vertex_offset = -1;
    IndexT abd_vertex_count = 0;
    IndexT abd_old_dof_offset = -1;
    IndexT abd_body_count = 0;
    muda::CBufferView<IndexT> abd_vertex_to_body;
    muda::CBufferView<IndexT> abd_body_is_fixed;
    if(abd_linear_subsystem && affine_body_dynamics && affine_body_vertex_reporter)
    {
        abd_vertex_offset = affine_body_vertex_reporter->vertex_offset();
        abd_vertex_count = affine_body_vertex_reporter->vertex_count();
        abd_old_dof_offset = abd_linear_subsystem->dof_offset();
        abd_vertex_to_body = affine_body_dynamics->v2b();
        abd_body_is_fixed = affine_body_dynamics->body_is_fixed();
        abd_body_count = static_cast<IndexT>(abd_body_is_fixed.size());
    }

    const auto old_to_chain = structured_info.old_to_chain();
    const auto shape = structured_info.shape();
    const StructuredVertexDescriptorCacheKey key{
        epoch,
        global_vertex_count,
        shape.horizon,
        shape.block_size,
        old_to_chain.data(),
        old_to_chain.size(),
        fem_vertex_offset,
        fem_vertex_count,
        fem_old_dof_offset,
        fem_vertex_is_fixed.data(),
        fem_vertex_is_fixed.size(),
        abd_vertex_offset,
        abd_vertex_count,
        abd_old_dof_offset,
        abd_body_count,
        abd_vertex_to_body.data(),
        abd_vertex_to_body.size(),
        abd_body_is_fixed.data(),
        abd_body_is_fixed.size()};

    if(structured_vertex_descriptor_key == key
       && structured_vertex_descriptors.size() == global_vertex_count)
    {
        // native_contact_descriptor_cache_hit_exports_view: plan building
        // happens before structured contact assembly, so the current
        // StructuredAssemblyInfo must receive the cached descriptor view here.
        structured_info.set_native_vertex_descriptors(
            structured_vertex_descriptors.view().as_const());
        return;
    }

    if(global_vertex_count > structured_vertex_descriptors.capacity())
    {
        const SizeT reserve_size =
            static_cast<SizeT>(static_cast<Float>(global_vertex_count) * reserve_ratio)
            + 1;
        muda::BufferLaunch(structured_info.stream())
            .reserve(structured_vertex_descriptors, reserve_size);
    }
    muda::BufferLaunch(structured_info.stream())
        .resize(structured_vertex_descriptors, global_vertex_count);

    rebuild_socu_native_vertex_descriptors(structured_info.stream(),
                                           structured_vertex_descriptors.view(),
                                           old_to_chain,
                                           shape.horizon,
                                           shape.block_size,
                                           epoch,
                                           fem_vertex_offset,
                                           fem_vertex_count,
                                           fem_old_dof_offset,
                                           fem_vertex_is_fixed,
                                           abd_vertex_offset,
                                           abd_vertex_count,
                                           abd_old_dof_offset,
                                           abd_body_count,
                                           abd_vertex_to_body,
                                           abd_body_is_fixed);
    structured_vertex_descriptor_key = key;
    structured_vertex_descriptor_epoch = epoch;
    structured_info.set_native_vertex_descriptors(
        structured_vertex_descriptors.view().as_const());
}

void GlobalDyTopoEffectManager::Impl::
    build_socu_contact_assembly_plan_m2(
        SocuContactAssemblyPlan&            plan,
        SocuContactAssemblyPlanM2Workspace& workspace,
        const SocuVertexSidePlanKey&        side_key,
        const SocuContactProgramPlanKey&    program_key,
        muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
        StructuredContactOffbandPolicy offband_policy,
        SocuVertexSideCoverageMode     coverage_mode,
        bool                            build_hot_block_plan,
        SocuContactExecutionStrategy    hot_block_strategy,
        SizeT                           hot_block_threshold,
        cudaStream_t                   stream)
{
    SocuContactAssemblyPlanM2BuildInput input;
    input.side_key = side_key;
    input.program_key = program_key;
    input.vertex_descriptors = vertex_descriptors;
    input.offband_policy = offband_policy;
    input.side_coverage_mode = coverage_mode;
    input.build_hot_block_plan = build_hot_block_plan;
    input.hot_block_strategy = hot_block_strategy;
    input.hot_block_threshold = hot_block_threshold;
    input.stream = stream;

    std::vector<SocuContactM2SourceInput> sources;
    sources.reserve(16);
    SizeT reporter_id = 0;
    SizeT source_id = 0;

    auto push_source = [&](std::uint32_t reporter,
                           SocuContactModelKind model,
                           SocuContactFamily family,
                           auto view)
    {
        SocuContactM2SourceInput source;
        source.source_id = static_cast<SocuContactSourceId>(source_id++);
        source.reporter_id = reporter;
        source.model = model;
        source.family = family;
        if constexpr(std::is_same_v<std::decay_t<decltype(view)>,
                                    muda::CBufferView<Vector4i>>)
        {
            source.stencil_size = 4;
            source.stencil4 = view;
        }
        else if constexpr(std::is_same_v<std::decay_t<decltype(view)>,
                                         muda::CBufferView<Vector3i>>)
        {
            source.stencil_size = 3;
            source.stencil3 = view;
        }
        else
        {
            source.stencil_size = 2;
            source.stencil2 = view;
        }
        sources.push_back(source);
    };

    for(auto&& reporter : dytopo_effect_reporters.view())
    {
        if(!has_flags(EnergyComponentFlags::Contact, reporter->component_flags()))
            continue;

        const auto current_reporter_id =
            static_cast<std::uint32_t>(reporter_id++);

        if(auto* normal = dynamic_cast<SimplexNormalContact*>(reporter))
        {
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexNormal,
                        SocuContactFamily::PT,
                        normal->PTs());
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexNormal,
                        SocuContactFamily::EE,
                        normal->EEs());
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexNormal,
                        SocuContactFamily::PE,
                        normal->PEs());
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexNormal,
                        SocuContactFamily::PP,
                        normal->PPs());
            continue;
        }
        if(auto* friction = dynamic_cast<SimplexFrictionalContact*>(reporter))
        {
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexFrictional,
                        SocuContactFamily::PT,
                        friction->PTs());
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexFrictional,
                        SocuContactFamily::EE,
                        friction->EEs());
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexFrictional,
                        SocuContactFamily::PE,
                        friction->PEs());
            push_source(current_reporter_id,
                        SocuContactModelKind::SimplexFrictional,
                        SocuContactFamily::PP,
                        friction->PPs());
            continue;
        }
        if(auto* normal = dynamic_cast<VertexHalfPlaneNormalContact*>(reporter))
        {
            push_source(current_reporter_id,
                        SocuContactModelKind::VertexHalfPlaneNormal,
                        SocuContactFamily::PH,
                        normal->PHs());
            continue;
        }
        if(auto* friction = dynamic_cast<VertexHalfPlaneFrictionalContact*>(reporter))
        {
            push_source(current_reporter_id,
                        SocuContactModelKind::VertexHalfPlaneFrictional,
                        SocuContactFamily::PH,
                        friction->PHs());
            continue;
        }

        throw SimSystemException{fmt::format(
            "socu_native_contact_plan_unsupported_reporter: reporter '{}' is "
            "a contact reporter but does not expose an M2 native contact source",
            reporter->name())};
    }

    input.sources = span<const SocuContactM2SourceInput>{sources};
    ::uipc::backend::cuda_mixed::build_socu_contact_assembly_plan_m2(
        plan,
        workspace,
        input);
}

SizeT GlobalDyTopoEffectManager::Impl::contact_set_signature()
{
    SizeT signature = ContactSignatureFnvOffset;

    SizeT reporter_index = 0;
    for(auto&& reporter : dytopo_effect_reporters.view())
    {
        if(!has_flags(EnergyComponentFlags::Contact, reporter->component_flags()))
            continue;

        GradientHessianExtentInfo extent_info;
        extent_info.m_gradient_only = false;
        reporter->report_gradient_hessian_extent(extent_info);

        mix_contact_signature(signature, reporter_index++);

        if(auto* normal = dynamic_cast<SimplexNormalContact*>(reporter))
        {
            mix_contact_vector_view(signature, SizeT{0x1001}, normal->PTs());
            mix_contact_vector_view(signature, SizeT{0x1002}, normal->EEs());
            mix_contact_vector_view(signature, SizeT{0x1003}, normal->PEs());
            mix_contact_vector_view(signature, SizeT{0x1004}, normal->PPs());
            continue;
        }
        if(auto* friction = dynamic_cast<SimplexFrictionalContact*>(reporter))
        {
            mix_contact_vector_view(signature, SizeT{0x2001}, friction->PTs());
            mix_contact_vector_view(signature, SizeT{0x2002}, friction->EEs());
            mix_contact_vector_view(signature, SizeT{0x2003}, friction->PEs());
            mix_contact_vector_view(signature, SizeT{0x2004}, friction->PPs());
            continue;
        }
        if(auto* normal = dynamic_cast<VertexHalfPlaneNormalContact*>(reporter))
        {
            mix_contact_vector_view(signature, SizeT{0x3001}, normal->PHs());
            continue;
        }
        if(auto* friction = dynamic_cast<VertexHalfPlaneFrictionalContact*>(reporter))
        {
            mix_contact_vector_view(signature, SizeT{0x4001}, friction->PHs());
            continue;
        }

        mix_contact_signature(signature, extent_info.m_gradient_count);
        mix_contact_signature(signature, extent_info.m_hessian_count);
    }
    mix_contact_signature(signature, reporter_index);
    return signature;
}

SocuContactTopologyStamp GlobalDyTopoEffectManager::Impl::contact_topology_stamp(
    cudaStream_t stream)
{
    SocuContactTopologyStamp stamp = socu_contact_topology_make_stamp_seed();
    socu_contact_topology_hash_reset(contact_topology_hash_workspace, stream);

    SizeT reporter_id = 0;
    SizeT source_id = 0;
    for(auto&& reporter : dytopo_effect_reporters.view())
    {
        if(!has_flags(EnergyComponentFlags::Contact, reporter->component_flags()))
            continue;

        const SizeT current_reporter_id = reporter_id++;

        auto register_view = [&](SocuContactSourceFamily family, auto view, SizeT& count)
        {
            count += view.size();
            socu_contact_topology_mix_view(stamp,
                                           contact_topology_hash_workspace,
                                           stream,
                                           current_reporter_id,
                                           source_id++,
                                           family,
                                           view);
        };

        if(auto* normal = dynamic_cast<SimplexNormalContact*>(reporter))
        {
            register_view(SocuContactSourceFamily::SimplexNormalPT,
                          normal->PTs(),
                          stamp.counts.simplex_normal_pt);
            register_view(SocuContactSourceFamily::SimplexNormalEE,
                          normal->EEs(),
                          stamp.counts.simplex_normal_ee);
            register_view(SocuContactSourceFamily::SimplexNormalPE,
                          normal->PEs(),
                          stamp.counts.simplex_normal_pe);
            register_view(SocuContactSourceFamily::SimplexNormalPP,
                          normal->PPs(),
                          stamp.counts.simplex_normal_pp);
            continue;
        }
        if(auto* friction = dynamic_cast<SimplexFrictionalContact*>(reporter))
        {
            register_view(SocuContactSourceFamily::SimplexFrictionPT,
                          friction->PTs(),
                          stamp.counts.simplex_friction_pt);
            register_view(SocuContactSourceFamily::SimplexFrictionEE,
                          friction->EEs(),
                          stamp.counts.simplex_friction_ee);
            register_view(SocuContactSourceFamily::SimplexFrictionPE,
                          friction->PEs(),
                          stamp.counts.simplex_friction_pe);
            register_view(SocuContactSourceFamily::SimplexFrictionPP,
                          friction->PPs(),
                          stamp.counts.simplex_friction_pp);
            continue;
        }
        if(auto* normal = dynamic_cast<VertexHalfPlaneNormalContact*>(reporter))
        {
            register_view(SocuContactSourceFamily::HalfPlaneNormalPH,
                          normal->PHs(),
                          stamp.counts.half_plane_normal_ph);
            continue;
        }
        if(auto* friction = dynamic_cast<VertexHalfPlaneFrictionalContact*>(reporter))
        {
            register_view(SocuContactSourceFamily::HalfPlaneFrictionPH,
                          friction->PHs(),
                          stamp.counts.half_plane_friction_ph);
            continue;
        }

        GradientHessianExtentInfo extent_info;
        extent_info.m_gradient_only = false;
        reporter->report_gradient_hessian_extent(extent_info);
        socu_contact_topology_mix_unknown_source(stamp,
                                                 current_reporter_id,
                                                 source_id++,
                                                 extent_info.m_hessian_count);
    }

    socu_contact_topology_finalize_metadata(stamp, reporter_id);

    if(stamp.source_count != 0)
    {
        const auto device_hash =
            socu_contact_topology_hash_finish(contact_topology_hash_workspace, stream);
        socu_contact_topology_mix_device_hash(stamp, device_hash);
    }

    return contact_topology_stamp_cache.update(stamp);
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceTripletMatrix<GlobalDyTopoEffectManager::StoreScalar, 3>& m,
    SizeT                                                                  size)
{
    if(size > m.triplet_capacity())
    {
        m.reserve_triplets(size * reserve_ratio);
    }
    m.resize_triplets(size);
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    muda::DeviceDoubletVector<GlobalDyTopoEffectManager::StoreScalar, 3>& v,
    SizeT                                                                  size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets(size * reserve_ratio);
    }
    v.resize_doublets(size);
}
}  // namespace uipc::backend::cuda_mixed


namespace uipc::backend::cuda_mixed
{
void GlobalDyTopoEffectManager::init()
{
    m_impl.init(world());
}

void GlobalDyTopoEffectManager::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::assemble_structured_hessian(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    m_impl.assemble_structured_hessian(info);
}

void GlobalDyTopoEffectManager::ensure_structured_vertex_descriptors(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    m_impl.ensure_structured_vertex_descriptors(info);
}

void GlobalDyTopoEffectManager::
    build_socu_contact_assembly_plan_m2(
        SocuContactAssemblyPlan&            plan,
        SocuContactAssemblyPlanM2Workspace& workspace,
        const SocuVertexSidePlanKey&        side_key,
        const SocuContactProgramPlanKey&    program_key,
        muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
        StructuredContactOffbandPolicy offband_policy,
        SocuVertexSideCoverageMode     coverage_mode,
        bool                            build_hot_block_plan,
        SocuContactExecutionStrategy    hot_block_strategy,
        SizeT                           hot_block_threshold,
        cudaStream_t                   stream)
{
    m_impl.build_socu_contact_assembly_plan_m2(
        plan,
        workspace,
        side_key,
        program_key,
        vertex_descriptors,
        offband_policy,
        coverage_mode,
        build_hot_block_plan,
        hot_block_strategy,
        hot_block_threshold,
        stream);
}

SizeT GlobalDyTopoEffectManager::contact_set_signature()
{
    return m_impl.contact_set_signature();
}

SocuContactTopologyStamp GlobalDyTopoEffectManager::contact_topology_stamp(
    cudaStream_t stream)
{
    return m_impl.contact_topology_stamp(stream);
}

void GlobalDyTopoEffectManager::compute_dytopo_effect()
{
    ComputeDyTopoEffectInfo info;
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::add_reporter(DyTopoEffectReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    auto flag = reporter->component_flags();
    UIPC_ASSERT(is_valid_flag(flag),
                "reporter component_flags() is not valid single flag, it's {}",
                enum_flags_name(flag));
    m_impl.dytopo_effect_reporters.register_sim_system(*reporter);

    // classify into contact / non-contact
    if(reporter->component_flags() == EnergyComponentFlags::Contact)
    {
        m_impl.contact_reporters.register_sim_system(*reporter);
    }
    else
    {
        m_impl.non_contact_reporters.register_sim_system(*reporter);
    }
}

void GlobalDyTopoEffectManager::add_receiver(DyTopoEffectReceiver* receiver)
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.dytopo_effect_receivers.register_sim_system(*receiver);
}
}  // namespace uipc::backend::cuda_mixed
