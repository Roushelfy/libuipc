#pragma once
#include <algorithm/matrix_converter.h>
#include <linear_system/diag_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <utils/assembly_sink.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::cuda_mixed
{
class FiniteElementKinetic;
class FEMDyTopoEffectReceiver;
class FEMLinearSubsystemReporter;
class FEMLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;
    using StoreScalar = GlobalLinearSystem::StoreScalar;

    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool                                     gradient_only,
                                   muda::DoubletVectorView<StoreScalar, 3>  gradients,
                                   muda::TripletMatrixView<StoreScalar, 3, 3> hessians,
                                   Float                                    dt,
                                   StructuredDeviceAssemblySink<
                                       StoreScalar,
                                       GlobalLinearSystem::SolveScalar> structured_sink = {},
                                   IndexT old_dof_offset = 0,
                                   muda::CBufferView<IndexT> fixed_vertices = {},
                                   bool identity_fixed_diagonal = false,
                                   bool write_gradients = true) noexcept
            : m_gradient_only(gradient_only)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_dt(dt)
            , m_structured_sink(structured_sink)
            , m_old_dof_offset(old_dof_offset)
            , m_fixed_vertices(fixed_vertices)
            , m_identity_fixed_diagonal(identity_fixed_diagonal)
            , m_write_gradients(write_gradients)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        auto gradients() const noexcept
        {
            return structured_assembly() && !m_write_gradients
                       ? muda::DoubletVectorView<StoreScalar, 3>{}
                       : m_gradients;
        }
        auto hessians() const noexcept
        {
            return structured_assembly()
                       ? muda::TripletMatrixView<StoreScalar, 3, 3>{}
                       : m_hessians;
        }
        bool structured_assembly() const noexcept
        {
            return m_structured_sink.valid();
        }
        auto structured_sink() const noexcept { return m_structured_sink; }
        IndexT old_dof_offset() const noexcept { return m_old_dof_offset; }
        auto fixed_vertices() const noexcept { return m_fixed_vertices; }
        bool identity_fixed_diagonal() const noexcept
        {
            return m_identity_fixed_diagonal;
        }
        bool write_gradients() const noexcept { return m_write_gradients; }
        auto sink() const noexcept
        {
            auto hessian_view =
                structured_assembly() ? muda::TripletMatrixView<StoreScalar, 3, 3>{}
                                      : m_hessians;
            return LocalAssemblySink<StoreScalar, GlobalLinearSystem::SolveScalar, 3>{
                gradients(),
                hessian_view,
                m_gradient_only,
                m_structured_sink,
                m_old_dof_offset,
                m_fixed_vertices,
                m_identity_fixed_diagonal,
                m_write_gradients};
        }
        auto dt() const noexcept { return m_dt; }

      private:
        bool                                     m_gradient_only = false;
        muda::DoubletVectorView<StoreScalar, 3>  m_gradients;
        muda::TripletMatrixView<StoreScalar, 3, 3> m_hessians;
        Float                                    m_dt = 0.0;
        StructuredDeviceAssemblySink<StoreScalar, GlobalLinearSystem::SolveScalar> m_structured_sink;
        IndexT                                  m_old_dof_offset = 0;
        muda::CBufferView<IndexT>               m_fixed_vertices;
        bool                                    m_identity_fixed_diagonal = false;
        bool                                    m_write_gradients = true;
    };

    class ReportExtentInfo
    {
      public:
        // DoubletVector3 count
        void gradient_count(SizeT size);
        // TripletMatrix3x3 count
        void hessian_count(SizeT size);
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class FEMLinearSubsystem;
        friend class FEMLinearSubsystemReporter;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
        bool  m_gradient_only  = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    class AssembleInfo
    {
      public:
        AssembleInfo(Impl*                                impl,
                     IndexT                               index,
                     GlobalLinearSystem::TripletMatrixView hessians,
                     bool                                 gradient_only,
                     StructuredDeviceAssemblySink<
                         StoreScalar,
                         GlobalLinearSystem::SolveScalar> structured_sink = {},
                     IndexT old_dof_offset = 0,
                     muda::CBufferView<IndexT> fixed_vertices = {},
                     bool identity_fixed_diagonal = false,
                     bool write_gradients = true) noexcept
            : m_impl(impl)
            , m_index(index)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
            , m_structured_sink(structured_sink)
            , m_old_dof_offset(old_dof_offset)
            , m_fixed_vertices(fixed_vertices)
            , m_identity_fixed_diagonal(identity_fixed_diagonal)
            , m_write_gradients(write_gradients)
        {
        }

        muda::DoubletVectorView<StoreScalar, 3> gradients() const;
        GlobalLinearSystem::TripletMatrixView hessians() const;
        Float                                dt() const noexcept;
        bool                                 gradient_only() const noexcept;
        bool structured_assembly() const noexcept
        {
            return m_structured_sink.valid();
        }
        auto structured_sink() const noexcept { return m_structured_sink; }
        IndexT old_dof_offset() const noexcept { return m_old_dof_offset; }
        auto fixed_vertices() const noexcept { return m_fixed_vertices; }
        bool identity_fixed_diagonal() const noexcept
        {
            return m_identity_fixed_diagonal;
        }
        bool write_gradients() const noexcept { return m_write_gradients; }
        auto sink() const noexcept
        {
            auto hessian_view =
                structured_assembly() ? GlobalLinearSystem::TripletMatrixView{}
                                      : hessians();
            return LocalAssemblySink<StoreScalar, GlobalLinearSystem::SolveScalar, 3>{
                gradients(),
                hessian_view,
                m_gradient_only,
                m_structured_sink,
                m_old_dof_offset,
                m_fixed_vertices,
                m_identity_fixed_diagonal,
                m_write_gradients};
        }

      private:
        friend class FEMLinearSubsystem;

        Impl*                                m_impl          = nullptr;
        IndexT                               m_index         = ~0;
        GlobalLinearSystem::TripletMatrixView m_hessians;
        bool                                 m_gradient_only = false;
        StructuredDeviceAssemblySink<StoreScalar, GlobalLinearSystem::SolveScalar> m_structured_sink;
        IndexT                               m_old_dof_offset = 0;
        muda::CBufferView<IndexT>            m_fixed_vertices;
        bool                                 m_identity_fixed_diagonal = false;
        bool                                 m_write_gradients = true;
    };

    class Impl
    {
      public:
        void init();
        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);

        void assemble(GlobalLinearSystem::DiagInfo& info);
        void _assemble_kinetic(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_reporters(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_dytopo_effect(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);


        void accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void retrieve_solution(GlobalLinearSystem::SolutionInfo& info);

        SimEngine* sim_engine = nullptr;

        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl&         fem() noexcept
        {
            return finite_element_method->m_impl;
        }

        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

        SimSystemSlot<FEMDyTopoEffectReceiver> dytopo_effect_receiver;
        SimSystemSlotCollection<FEMLinearSubsystemReporter> reporters;

        SimSystemSlot<FiniteElementKinetic> kinetic;


        Float dt            = 0.0;
        Float reserve_ratio = 1.5;

        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        muda::DeviceDoubletVector<StoreScalar, 3> kinetic_gradients;
        muda::DeviceDoubletVector<StoreScalar, 3> reporter_gradients;

        void loose_resize_entries(muda::DeviceDoubletVector<StoreScalar, 3>& v, SizeT size);
        void assemble_structured(GlobalLinearSystem::StructuredAssemblyInfo& info);
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(DiagLinearSubsystem::InitInfo& info) override;
    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual bool do_supports_structured_assembly() const override;
    virtual void do_assemble_structured(GlobalLinearSystem::StructuredAssemblyInfo& info) override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;
    virtual U64 get_uid() const noexcept override;

    friend class FEMLinearSubsystemReporter;
    void add_reporter(FEMLinearSubsystemReporter* reporter);

    friend class FiniteElementKinetic;
    void add_kinetic(FiniteElementKinetic* kinetic);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda_mixed
