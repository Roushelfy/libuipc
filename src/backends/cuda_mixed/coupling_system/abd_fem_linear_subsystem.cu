#include <linear_system/off_diag_linear_subsystem.h>
#include <coupling_system/abd_fem_dytopo_effect_receiver.h>
#include <coupling_system/fem_abd_dytopo_effect_receiver.h>
#include <affine_body/abd_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <affine_body/affine_body_dynamics.h>
#include <finite_element/finite_element_method.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <kernel_cout.h>
#include <utils/matrix_unpacker.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class ABDFEMLinearSubsystem final : public OffDiagLinearSubsystem
{
  public:
    using OffDiagLinearSubsystem::OffDiagLinearSubsystem;
    using StoreScalar = GlobalLinearSystem::StoreScalar;

    SimSystemSlot<GlobalLinearSystem> global_linear_system;

    SimSystemSlot<ABDFEMDyTopoEffectReceiver> abd_fem_dytopo_effect_receiver;

    SimSystemSlot<ABDLinearSubsystem> abd_linear_subsystem;
    SimSystemSlot<FEMLinearSubsystem> fem_linear_subsystem;

    SimSystemSlot<AffineBodyDynamics>  affine_body_dynamics;
    SimSystemSlot<FiniteElementMethod> finite_element_method;

    SimSystemSlot<AffineBodyVertexReporter>    affine_body_vertex_reporter;
    SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

    virtual void do_build(BuildInfo& info) override
    {
        global_linear_system = require<GlobalLinearSystem>();

        abd_fem_dytopo_effect_receiver = require<ABDFEMDyTopoEffectReceiver>();

        abd_linear_subsystem = require<ABDLinearSubsystem>();
        fem_linear_subsystem = require<FEMLinearSubsystem>();

        affine_body_dynamics  = require<AffineBodyDynamics>();
        finite_element_method = require<FiniteElementMethod>();

        affine_body_vertex_reporter    = require<AffineBodyVertexReporter>();
        finite_element_vertex_reporter = require<FiniteElementVertexReporter>();

        info.connect(abd_linear_subsystem.view(), fem_linear_subsystem.view());
    }

    virtual void report_extent(GlobalLinearSystem::OffDiagExtentInfo& info) override
    {
        if(!abd_fem_dytopo_effect_receiver)
        {
            info.extent(0, 0);
            return;
        }

        // ABD-FEM Hessian: H12x3
        auto abd_fem_dytopo_effect_count =
            abd_fem_dytopo_effect_receiver->hessians().triplet_count();
        auto abd_fem_H3x3_count = abd_fem_dytopo_effect_count * 4;

        info.extent(abd_fem_H3x3_count, 0);
    }

    virtual void assemble(GlobalLinearSystem::OffDiagInfo& info) override
    {
        using namespace muda;
        using Alu = ActivePolicy::AluScalar;

        auto count = abd_fem_dytopo_effect_receiver->hessians().triplet_count();

        if(count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(count,
                       [v2b = affine_body_dynamics->v2b().viewer().name("v2b"),
                        Js  = affine_body_dynamics->Js().viewer().name("Js"),
                        body_is_fixed =
                            affine_body_dynamics->body_is_fixed().viewer().name("body_is_fixed"),
                        vertex_is_fixed =
                            finite_element_method->is_fixed().viewer().name("vertex_is_fixed"),
                        L = info.lr_hessian().viewer().name("L"),
                        abd_fem_dytopo_effect =
                            abd_fem_dytopo_effect_receiver->hessians().viewer().name("abd_fem_dytopo_effect"),
                        abd_point_offset = affine_body_vertex_reporter->vertex_offset(),
                        fem_point_offset =
                            finite_element_vertex_reporter->vertex_offset()] __device__(int I) mutable
                       {
                           {
                               // global vertex indices
                               auto&& [gI_abd_v, gJ_fem_v, H3x3] =
                                   abd_fem_dytopo_effect(I);

                               MUDA_ASSERT(abd_point_offset <= fem_point_offset,
                                           "We assume ABD vertices are before FEM vertices, "
                                           "but got abd_point_offset=%d, fem_point_offset=%d",
                                           abd_point_offset,
                                           fem_point_offset);

                               auto I_abd_v = gI_abd_v - abd_point_offset;
                               auto J_fem_v = gJ_fem_v - fem_point_offset;

                               auto body_id = v2b(I_abd_v);

                               const ABDJacobi& J = Js(I_abd_v);

                               auto I4 = 4 * I;

                               using StoreScalar = GlobalLinearSystem::StoreScalar;
                               Eigen::Matrix<Alu, 12, 3> H_alu =
                                   J.to_mat().transpose().template cast<Alu>()
                                   * H3x3.template cast<Alu>();
                               auto H = downcast_hessian<StoreScalar>(H_alu);

                               if(body_is_fixed(body_id) || vertex_is_fixed(J_fem_v))
                                   H.setZero();

                               //for(int k = 0; k < 4; ++k)
                               //{
                               //    L(I4 + k).write(4 * body_id + k,  // abd
                               //                    J_fem_v,          // fem
                               //                    H.block<3, 3>(3 * k, 0));
                               //}

                               TripletMatrixUnpacker TMU{L};

                               TMU.block<4, 1>(I4).write(4 * body_id, J_fem_v, H);
                           }
                       });
        }
    }

    virtual bool do_supports_structured_assembly() const override
    {
        return true;
    }

    virtual void do_assemble_structured(
        GlobalLinearSystem::StructuredAssemblyInfo& info) override
    {
        using namespace muda;
        using Alu = ActivePolicy::AluScalar;

        auto count = abd_fem_dytopo_effect_receiver->hessians().triplet_count();
        if(count == 0)
            return;

        muda::DeviceBuffer<IndexT> structured_counts;
        muda::BufferView<IndexT>   structured_counts_view;
        if(info.report_counters_enabled())
        {
            structured_counts.resize(3);
            muda::BufferLaunch(info.stream()).fill<IndexT>(structured_counts.view(), 0);
            structured_counts_view = structured_counts.view();
        }

        auto sink = info.sink();
        const IndexT abd_old_dof_offset =
            static_cast<IndexT>(abd_linear_subsystem->dof_offset());
        const IndexT fem_old_dof_offset =
            static_cast<IndexT>(fem_linear_subsystem->dof_offset());

        ParallelFor(256, 0, info.stream())
            .file_line(__FILE__, __LINE__)
            .apply(count,
                   [sink,
                    abd_old_dof_offset,
                    fem_old_dof_offset,
                    v2b = affine_body_dynamics->v2b().viewer().name("v2b"),
                    Js  = affine_body_dynamics->Js().viewer().name("Js"),
                    body_is_fixed =
                        affine_body_dynamics->body_is_fixed().viewer().name("body_is_fixed"),
                    vertex_is_fixed =
                        finite_element_method->is_fixed().viewer().name("vertex_is_fixed"),
                    abd_fem_dytopo_effect =
                        abd_fem_dytopo_effect_receiver->hessians().viewer().name("abd_fem_dytopo_effect"),
                    counts = structured_counts_view,
                    abd_point_offset = affine_body_vertex_reporter->vertex_offset(),
                    fem_point_offset =
                        finite_element_vertex_reporter->vertex_offset()] __device__(int I) mutable
                   {
                       auto&& [gI_abd_v, gJ_fem_v, H3x3] =
                           abd_fem_dytopo_effect(I);

                       auto I_abd_v = gI_abd_v - abd_point_offset;
                       auto J_fem_v = gJ_fem_v - fem_point_offset;
                       auto body_id = v2b(I_abd_v);

                       const ABDJacobi& J = Js(I_abd_v);
                       Eigen::Matrix<Alu, 12, 3> H_alu =
                           J.to_mat().transpose().template cast<Alu>()
                           * H3x3.template cast<Alu>();

                       if(body_is_fixed(body_id) || vertex_is_fixed(J_fem_v))
                           H_alu.setZero();

                       const IndexT old_abd = abd_old_dof_offset + body_id * 12;
                       const IndexT old_fem = fem_old_dof_offset + J_fem_v * 3;

                       bool first_offdiag = false;
                       bool off_band      = false;
                       if(static_cast<SizeT>(old_abd) < sink.old_to_chain.size()
                          && static_cast<SizeT>(old_fem) < sink.old_to_chain.size())
                       {
                           const IndexT chain_abd =
                               sink.old_to_chain[static_cast<SizeT>(old_abd)];
                           const IndexT chain_fem =
                               sink.old_to_chain[static_cast<SizeT>(old_fem)];
                           if(chain_abd >= 0 && chain_fem >= 0)
                           {
                               const SizeT block_abd =
                                   static_cast<SizeT>(chain_abd) / sink.block_size;
                               const SizeT block_fem =
                                   static_cast<SizeT>(chain_fem) / sink.block_size;
                               const SizeT distance =
                                   block_abd > block_fem ? block_abd - block_fem
                                                         : block_fem - block_abd;
                               first_offdiag = distance == 1;
                               off_band      = distance > 1;
                           }
                       }

                       if(!off_band)
                       {
                           const auto H_store = downcast_hessian<StoreScalar>(H_alu);
                           sink.template add_dense_block_between_fixed<12, 3>(
                               old_abd,
                               old_fem,
                               H_store);
                           sink.template add_dense_block_between_fixed<3, 12>(
                               old_fem,
                               old_abd,
                               H_store.transpose());
                       }

                       if(counts.data() != nullptr)
                       {
                           if(off_band)
                               muda::atomic_add(counts.data(2), IndexT{72});
                           else if(first_offdiag)
                               muda::atomic_add(counts.data(1), IndexT{72});
                           else
                               muda::atomic_add(counts.data(0), IndexT{72});
                       }
                   });

        if(info.report_counters_enabled())
        {
            std::array<IndexT, 3> counts_host{};
            structured_counts.view().copy_to(counts_host.data());
            info.record_diag_writes(static_cast<SizeT>(counts_host[0]));
            info.record_first_offdiag_writes(static_cast<SizeT>(counts_host[1]));
            info.record_contact_band_stats(0,
                                           0,
                                           static_cast<SizeT>(counts_host[0]
                                                              + counts_host[1]),
                                           static_cast<SizeT>(counts_host[2]));
        }
    }
};

REGISTER_SIM_SYSTEM(ABDFEMLinearSubsystem);
}  // namespace uipc::backend::cuda_mixed
