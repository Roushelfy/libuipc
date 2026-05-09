#pragma once
#include <sim_system.h>
#include <functional>
#include <uipc/common/list.h>
#include <uipc/common/vector.h>
#include <muda/ext/linear_system.h>
#include <algorithm/matrix_converter.h>
#include <linear_system/assembly_mode.h>
#include <linear_system/structured_chain_provider.h>
#include <utils/assembly_sink.h>
#include <utils/offset_count_collection.h>
#include <utils/structured_contact_hessian_cache.h>
#include <utils/structured_contact_offband_policy.h>
#include <energy_component_flags.h>
#include <mixed_precision/policy.h>
#include <cuda_runtime_api.h>

namespace uipc::backend::cuda_mixed
{
// Define a simple POD to avoid constructing CUDA's built-in vector type with pmr allocators in host code
struct SizeT2
{
    SizeT x;
    SizeT y;
};

class DiagLinearSubsystem;
class OffDiagLinearSubsystem;
class LinearSolver;
class LocalPreconditioner;
class GlobalPreconditioner;
class GlobalDyTopoEffectManager;

class GlobalLinearSystem : public SimSystem
{
    static constexpr SizeT DoFBlockSize = 3;

  public:
    using SimSystem::SimSystem;
    using StoreScalar = ActivePolicy::StoreScalar;
    using PcgAuxScalar = ActivePolicy::PcgAuxScalar;
    using SolveScalar = ActivePolicy::SolveScalar;

    using TripletMatrixView = muda::TripletMatrixView<StoreScalar, 3>;
    using CBCOOMatrixView   = muda::CBCOOMatrixView<StoreScalar, 3>;
    using DenseVectorView   = muda::DenseVectorView<StoreScalar>;
    using CDenseVectorView  = muda::CDenseVectorView<StoreScalar>;
    using PcgDenseVectorView = muda::DenseVectorView<PcgAuxScalar>;
    using CPcgDenseVectorView = muda::CDenseVectorView<PcgAuxScalar>;
    using SolveDenseVectorView = muda::DenseVectorView<SolveScalar>;
    using CSolveDenseVectorView = muda::CDenseVectorView<SolveScalar>;
    using ComponentFlags    = EnergyComponentFlags;

    class Impl;

    enum class StructuredAssemblyPhase
    {
        ChainBase,
        Contact,
    };

    struct LineSearchFeedback
    {
        bool  accepted       = true;
        SizeT iteration_count = 0;
        bool  hit_max_iter   = false;
        Float accepted_alpha = 1.0;
    };

    class InitDofExtentInfo
    {
      public:
        void extent(SizeT dof_count) noexcept { m_dof_count = dof_count; }

      private:
        friend class Impl;
        SizeT m_dof_count = 0;
    };

    class InitDofInfo
    {
      public:
        IndexT dof_offset() const { return m_dof_offset; }
        IndexT dof_count() const { return m_dof_count; }

      private:
        friend class Impl;
        IndexT m_dof_offset = 0;
        IndexT m_dof_count  = 0;
    };

    class DiagExtentInfo
    {
      public:
        bool           gradient_only() const { return m_gradient_only; }
        ComponentFlags component_flags() const { return m_component_flags; }
        void extent(SizeT hessian_count, SizeT dof_count) noexcept;

      private:
        friend class Impl;
        ComponentFlags m_component_flags = ComponentFlags::All;
        SizeT          m_dof_count       = 0;
        SizeT          m_block_count     = 0;
        bool           m_gradient_only   = false;
    };

    class ComputeGradientInfo
    {
      public:
        /**
         * Output gradient vector view
         */
        void buffer_view(DenseVectorView grad) noexcept;
        /**
         * Specify which component to be taken into account during gradient computation
         * - Contact: only consider contact part
         * - Complement: only consider non-contact part
         */
        void flags(ComponentFlags component) noexcept;

      private:
        friend class Impl;
        DenseVectorView m_gradients;
        ComponentFlags               m_flags = ComponentFlags::All;
    };

    class DiagInfo
    {
      public:
        DiagInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        TripletMatrixView hessians() const { return m_hessians; }
        DenseVectorView   gradients() const { return m_gradients; }
        bool              gradient_only() const { return m_gradient_only; }
        ComponentFlags    component_flags() const { return m_component_flags; }

      private:
        friend class Impl;
        SizeT             m_index = ~0ull;
        TripletMatrixView m_hessians;
        DenseVectorView   m_gradients;
        bool              m_gradient_only   = false;
        ComponentFlags    m_component_flags = ComponentFlags::All;

        Impl* m_impl = nullptr;
    };

    class StructuredAssemblyInfo
    {
      public:
        StructuredAssemblyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        StructuredChainShape shape() const noexcept { return m_shape; }
        span<const StructuredDofSlot> dof_slots() const noexcept { return m_dof_slots; }
        CDenseVectorView b() const noexcept { return m_b; }
        cudaStream_t stream() const noexcept { return m_stream; }
        SizeT old_dof_offset() const noexcept { return m_old_dof_offset; }
        SizeT old_dof_count() const noexcept { return m_old_dof_count; }
        StructuredAssemblyPhase phase() const noexcept { return m_phase; }

        muda::BufferView<SolveScalar> diag() const noexcept { return m_diag; }
        muda::BufferView<SolveScalar> first_offdiag() const noexcept
        {
            return m_first_offdiag;
        }
        muda::BufferView<SolveScalar> rhs() const noexcept { return m_rhs; }
        muda::CBufferView<IndexT> old_to_chain() const noexcept
        {
            return m_old_to_chain;
        }
        muda::CBufferView<IndexT> chain_to_old() const noexcept
        {
            return m_chain_to_old;
        }
        muda::BufferView<IndexT> contact_counters() const noexcept
        {
            return m_contact_counters;
        }
        RuntimeOrderingCollector runtime_ordering_collector() const noexcept
        {
            return m_runtime_ordering_collector;
        }
        StructuredContactHessianCache<StoreScalar> contact_hessian_cache() const noexcept
        {
            return m_contact_hessian_cache;
        }
        StructuredContactOffbandPolicy contact_offband_policy() const noexcept
        {
            return m_contact_offband_policy;
        }
        SizeT contact_set_signature() const noexcept
        {
            return m_contact_set_signature;
        }
        IndexT descriptor_epoch() const noexcept { return m_descriptor_epoch; }
        bool report_counters_enabled() const noexcept
        {
            return m_contact_counters.data() != nullptr;
        }
        bool debug_timing_enabled() const noexcept { return m_debug_timing; }
        double chain_base_assembly_time_ms() const noexcept
        {
            return m_chain_base_assembly_time_ms;
        }
        double contact_assembly_time_ms() const noexcept
        {
            return m_contact_assembly_time_ms;
        }

        StructuredDeviceAssemblySink<StoreScalar, SolveScalar> sink() const noexcept
        {
            auto sink = StructuredDeviceAssemblySink<StoreScalar, SolveScalar>{
                m_diag,
                m_first_offdiag,
                m_old_to_chain,
                m_shape.horizon,
                m_shape.block_size,
                m_contact_counters,
                m_runtime_ordering_collector};
            if(m_phase == StructuredAssemblyPhase::ChainBase)
            {
                const auto native_view =
                    native_matrix_view(m_diag, m_first_offdiag, m_rhs);
                sink.matrix.native_matrix          = native_view;
                sink.matrix.native_dof_descriptors = m_native_dof_descriptors;
                sink.matrix.use_native_matrix =
                    m_native_chain_base_hessian_enabled;
                sink.matrix.compare_enabled =
                    m_chain_base_compare_enabled;
                sink.matrix.compare_uses_native_matrix =
                    m_chain_base_compare_uses_native;
                sink.matrix.compare_diag          = m_chain_base_compare_diag;
                sink.matrix.compare_first_offdiag =
                    m_chain_base_compare_first_offdiag;
                sink.matrix.compare_native_matrix =
                    native_matrix_view(m_chain_base_compare_diag,
                                       m_chain_base_compare_first_offdiag,
                                       m_chain_base_compare_rhs);
            }
            else if(m_phase == StructuredAssemblyPhase::Contact)
            {
                const auto native_view =
                    native_matrix_view(m_diag, m_first_offdiag, m_rhs);
                sink.matrix.native_matrix          = native_view;
                sink.matrix.native_dof_descriptors = m_native_dof_descriptors;
                sink.matrix.use_native_matrix =
                    m_native_contact_hessian_enabled;
                sink.matrix.compare_enabled =
                    m_contact_compare_enabled;
                sink.matrix.compare_uses_native_matrix =
                    m_contact_compare_uses_native;
                sink.matrix.compare_diag          = m_contact_compare_diag;
                sink.matrix.compare_first_offdiag =
                    m_contact_compare_first_offdiag;
                sink.matrix.compare_native_matrix =
                    native_matrix_view(m_contact_compare_diag,
                                       m_contact_compare_first_offdiag,
                                       m_contact_compare_rhs);
            }
            return sink;
        }

        void set_workspace(StructuredChainShape       shape,
                           span<const StructuredDofSlot> dof_slots,
                           muda::BufferView<SolveScalar> diag,
                           muda::BufferView<SolveScalar> first_offdiag,
                           muda::BufferView<SolveScalar> rhs,
                           muda::CBufferView<IndexT> old_to_chain,
                           muda::CBufferView<IndexT> chain_to_old,
                           cudaStream_t             stream) noexcept;
        void set_contact_counters(muda::BufferView<IndexT> counters) noexcept
        {
            m_contact_counters = counters;
        }
        void set_runtime_ordering_collector(RuntimeOrderingCollector collector) noexcept
        {
            m_runtime_ordering_collector = collector;
        }
        void set_contact_hessian_cache(
            StructuredContactHessianCache<StoreScalar> cache) noexcept
        {
            m_contact_hessian_cache = cache;
        }
        void set_contact_offband_policy(
            StructuredContactOffbandPolicy policy) noexcept
        {
            m_contact_offband_policy = policy;
        }
        void set_contact_set_signature(SizeT signature) noexcept
        {
            m_contact_set_signature = signature;
        }
        void set_descriptor_epoch(IndexT epoch) noexcept
        {
            m_descriptor_epoch = epoch;
        }
        void set_debug_timing(bool enabled) noexcept
        {
            m_debug_timing = enabled;
        }
        void set_phase(StructuredAssemblyPhase phase) noexcept
        {
            m_phase = phase;
        }
        void set_native_chain_base_hessian(
            bool enabled,
            muda::CBufferView<SocuNativeDofDescriptor> dof_descriptors) noexcept
        {
            m_native_chain_base_hessian_enabled = enabled;
            m_native_dof_descriptors = dof_descriptors;
        }
        void set_native_chain_base_compare_workspace(
            muda::BufferView<SolveScalar> compare_diag,
            muda::BufferView<SolveScalar> compare_first_offdiag,
            muda::BufferView<SolveScalar> compare_rhs,
            bool compare_uses_native) noexcept
        {
            m_chain_base_compare_diag          = compare_diag;
            m_chain_base_compare_first_offdiag = compare_first_offdiag;
            m_chain_base_compare_rhs           = compare_rhs;
            m_chain_base_compare_uses_native   = compare_uses_native;
            m_chain_base_compare_enabled =
                compare_diag.data() != nullptr;
        }
        void set_native_contact_hessian(
            bool enabled,
            muda::CBufferView<SocuNativeDofDescriptor> dof_descriptors) noexcept
        {
            m_native_contact_hessian_enabled = enabled;
            if(dof_descriptors.data() != nullptr)
                m_native_dof_descriptors = dof_descriptors;
        }
        void set_native_contact_compare_workspace(
            muda::BufferView<SolveScalar> compare_diag,
            muda::BufferView<SolveScalar> compare_first_offdiag,
            muda::BufferView<SolveScalar> compare_rhs,
            bool compare_uses_native) noexcept
        {
            m_contact_compare_diag          = compare_diag;
            m_contact_compare_first_offdiag = compare_first_offdiag;
            m_contact_compare_rhs           = compare_rhs;
            m_contact_compare_uses_native   = compare_uses_native;
            m_contact_compare_enabled =
                compare_diag.data() != nullptr;
        }

        void set_subsystem_extent(SizeT old_dof_offset, SizeT old_dof_count) noexcept;
        void record_diag_writes(SizeT count) noexcept { m_diag_write_count += count; }
        void record_first_offdiag_writes(SizeT count) noexcept
        {
            m_first_offdiag_write_count += count;
        }
        void record_contact_diag_writes(SizeT count) noexcept
        {
            m_contact_diag_write_count += count;
        }
        void record_contact_first_offdiag_writes(SizeT count) noexcept
        {
            m_contact_first_offdiag_write_count += count;
        }
        void record_contact_band_stats(SizeT near_contact_count,
                                       SizeT off_band_contact_count,
                                       SizeT near_contribution_count,
                                       SizeT off_band_contribution_count,
                                       SizeT diag_fallback_count = 0,
                                       SizeT lump_fallback_count = 0) noexcept
        {
            m_near_band_contact_count += near_contact_count;
            m_off_band_contact_count += off_band_contact_count;
            m_near_band_contribution_count += near_contribution_count;
            m_off_band_contribution_count += off_band_contribution_count;
            m_contact_diag_fallback_count += diag_fallback_count;
            m_contact_lump_fallback_count += lump_fallback_count;
        }
        void record_chain_base_assembly_time_ms(double elapsed_ms) noexcept
        {
            m_chain_base_assembly_time_ms += elapsed_ms;
        }
        void record_contact_assembly_time_ms(double elapsed_ms) noexcept
        {
            m_contact_assembly_time_ms += elapsed_ms;
        }
        SizeT diag_write_count() const noexcept { return m_diag_write_count; }
        SizeT first_offdiag_write_count() const noexcept
        {
            return m_first_offdiag_write_count;
        }
        SizeT contact_diag_write_count() const noexcept
        {
            return m_contact_diag_write_count;
        }
        SizeT contact_first_offdiag_write_count() const noexcept
        {
            return m_contact_first_offdiag_write_count;
        }
        SizeT near_band_contact_count() const noexcept
        {
            return m_near_band_contact_count;
        }
        SizeT off_band_contact_count() const noexcept
        {
            return m_off_band_contact_count;
        }
        SizeT near_band_contribution_count() const noexcept
        {
            return m_near_band_contribution_count;
        }
        SizeT off_band_contribution_count() const noexcept
        {
            return m_off_band_contribution_count;
        }
        SizeT contact_diag_fallback_count() const noexcept
        {
            return m_contact_diag_fallback_count;
        }
        SizeT contact_lump_fallback_count() const noexcept
        {
            return m_contact_lump_fallback_count;
        }
        bool configured() const noexcept { return m_configured; }

      private:
        friend class Impl;
        Impl* m_impl = nullptr;
        StructuredChainShape       m_shape;
        span<const StructuredDofSlot> m_dof_slots;
        CDenseVectorView           m_b;
        muda::BufferView<SolveScalar> m_diag;
        muda::BufferView<SolveScalar> m_first_offdiag;
        muda::BufferView<SolveScalar> m_rhs;
        muda::CBufferView<IndexT>  m_old_to_chain;
        muda::CBufferView<IndexT>  m_chain_to_old;
        muda::BufferView<IndexT>   m_contact_counters;
        RuntimeOrderingCollector   m_runtime_ordering_collector;
        StructuredContactHessianCache<StoreScalar> m_contact_hessian_cache;
        StructuredContactOffbandPolicy m_contact_offband_policy =
            StructuredContactOffbandPolicy::Drop;
        SizeT                      m_contact_set_signature = 0;
        IndexT                     m_descriptor_epoch = 0;
        cudaStream_t               m_stream = cudaStreamLegacy;
        SizeT                      m_old_dof_offset = 0;
        SizeT                      m_old_dof_count  = 0;
        SizeT                      m_diag_write_count = 0;
        SizeT                      m_first_offdiag_write_count = 0;
        SizeT                      m_contact_diag_write_count = 0;
        SizeT                      m_contact_first_offdiag_write_count = 0;
        SizeT                      m_near_band_contact_count = 0;
        SizeT                      m_off_band_contact_count = 0;
        SizeT                      m_near_band_contribution_count = 0;
        SizeT                      m_off_band_contribution_count = 0;
        SizeT                      m_contact_diag_fallback_count = 0;
        SizeT                      m_contact_lump_fallback_count = 0;
        double                     m_chain_base_assembly_time_ms = 0.0;
        double                     m_contact_assembly_time_ms = 0.0;
        StructuredAssemblyPhase    m_phase = StructuredAssemblyPhase::ChainBase;
        muda::CBufferView<SocuNativeDofDescriptor> m_native_dof_descriptors;
        bool                       m_native_chain_base_hessian_enabled = false;
        bool                       m_native_contact_hessian_enabled = false;
        bool                       m_debug_timing = false;
        muda::BufferView<SolveScalar> m_chain_base_compare_diag;
        muda::BufferView<SolveScalar> m_chain_base_compare_first_offdiag;
        muda::BufferView<SolveScalar> m_chain_base_compare_rhs;
        bool                       m_chain_base_compare_enabled = false;
        bool                       m_chain_base_compare_uses_native = false;
        muda::BufferView<SolveScalar> m_contact_compare_diag;
        muda::BufferView<SolveScalar> m_contact_compare_first_offdiag;
        muda::BufferView<SolveScalar> m_contact_compare_rhs;
        bool                       m_contact_compare_enabled = false;
        bool                       m_contact_compare_uses_native = false;
        bool                       m_configured = false;

        SocuNativeMatrixView<SolveScalar> native_matrix_view(
            muda::BufferView<SolveScalar> diag,
            muda::BufferView<SolveScalar> first_offdiag,
            muda::BufferView<SolveScalar> rhs) const noexcept
        {
            const SizeT block_elements = m_shape.block_size * m_shape.block_size;
            const SizeT offdiag_block_count =
                block_elements == 0 ? SizeT{0}
                                    : first_offdiag.size() / block_elements;
            return SocuNativeMatrixView<SolveScalar>{
                diag,
                first_offdiag,
                rhs,
                {},
                m_shape.horizon,
                m_shape.block_size,
                m_shape.nrhs,
                m_shape.horizon > 0 ? m_shape.horizon - 1 : SizeT{0},
                offdiag_block_count};
        }
    };

    class OffDiagExtentInfo
    {
      public:
        void extent(SizeT lr_hessian_block_count, SizeT rl_hessian_block_count) noexcept;

      private:
        friend class Impl;
        SizeT m_lr_block_count = 0;
        SizeT m_rl_block_count = 0;
    };

    class OffDiagInfo
    {
      public:
        OffDiagInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        TripletMatrixView lr_hessian() const { return m_lr_hessian; }
        TripletMatrixView rl_hessian() const { return m_rl_hessian; }

      private:
        friend class Impl;
        SizeT             m_index = ~0ull;
        TripletMatrixView m_lr_hessian;
        TripletMatrixView m_rl_hessian;
        Impl*             m_impl = nullptr;
    };

    class AssemblyInfo
    {
      public:
        AssemblyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        CBCOOMatrixView A() const;

      protected:
        friend class Impl;
        Impl* m_impl = nullptr;
    };

    class GlobalPreconditionerAssemblyInfo : public AssemblyInfo
    {
      public:
        using AssemblyInfo::AssemblyInfo;
    };

    class LocalPreconditionerAssemblyInfo : public AssemblyInfo
    {
      public:
        LocalPreconditionerAssemblyInfo(Impl* impl, SizeT index) noexcept
            : AssemblyInfo(impl)
            , m_index(index)
        {
        }

        SizeT dof_offset() const;
        SizeT dof_count() const;

      private:
        SizeT m_index;
    };

    class ApplyPreconditionerInfo
    {
      public:
        ApplyPreconditionerInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        PcgDenseVectorView  z() { return m_z; }
        CPcgDenseVectorView r() { return m_r; }
        muda::CVarView<IndexT> converged() { return m_converged; }

      private:
        friend class Impl;
        PcgDenseVectorView  m_z;
        CPcgDenseVectorView m_r;
        muda::CVarView<IndexT> m_converged;
        Impl*                  m_impl = nullptr;
    };

    class AccuracyInfo
    {
      public:
        AccuracyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        CPcgDenseVectorView r() const { return m_r; }

        void satisfied(bool statisfied) { m_statisfied = statisfied; }

      private:
        friend class Impl;
        CPcgDenseVectorView m_r;
        Impl*            m_impl       = nullptr;
        bool             m_statisfied = true;
    };

    class SolvingInfo
    {
      public:
        SolvingInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        SolveDenseVectorView x() { return m_x; }
        CDenseVectorView b() { return m_b; }
        void iter_count(SizeT iter_count) { m_iter_count = iter_count; }
        SizeT iter_count() const { return m_iter_count; }

      private:
        friend class Impl;
        SolveDenseVectorView m_x;
        CDenseVectorView m_b;
        SizeT            m_iter_count = 0;
        Impl*            m_impl       = nullptr;
    };

    class SolutionInfo
    {
      public:
        SolutionInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        CSolveDenseVectorView solution() { return m_solution; }

      private:
        friend class Impl;
        CSolveDenseVectorView m_solution;
        Impl*            m_impl = nullptr;
    };

  private:
    class LinearSubsytemInfo
    {
      public:
        bool  is_diag     = false;
        SizeT local_index = ~0ull;
        SizeT index       = ~0ull;
    };

  public:
    class Impl
    {
      public:
        void init();

        void build_linear_system();
        void _validate_structured_chain_subsystems();
        bool _update_subsystem_extent(bool needs_full_sparse_A);
        void _assemble_gradient_vector();
        void _assemble_linear_system();
        void _assemble_structured_chain();
        void _assemble_preconditioner();
        void solve_linear_system();
        void distribute_solution();

        Float reserve_ratio = 1.1;

        std::vector<LinearSubsytemInfo> subsystem_infos;

        OffsetCountCollection<IndexT> diag_dof_offsets_counts;
        OffsetCountCollection<IndexT> subsystem_triplet_offsets_counts;

        std::vector<SizeT2> off_diag_lr_triplet_counts;


        // Containers
        SimSystemSlotCollection<DiagLinearSubsystem>    diag_subsystems;
        SimSystemSlotCollection<OffDiagLinearSubsystem> off_diag_subsystems;
        SimSystemSlotCollection<LocalPreconditioner>    local_preconditioners;

        SimSystemSlotCollection<LinearSolver> linear_solvers;
        SimSystemSlot<GlobalPreconditioner> global_preconditioner;
        SimSystemSlot<GlobalDyTopoEffectManager> global_dytopo_effect_manager;

        LinearSolver* selected_linear_solver = nullptr;

        // Linear System
        muda::LinearSystemContext                ctx;
        muda::DeviceDenseVector<SolveScalar>     x;
        muda::DeviceDenseVector<StoreScalar>     b;
        muda::DeviceTripletMatrix<StoreScalar, 3> triplet_A;
        muda::DeviceBCOOMatrix<StoreScalar, 3>    bcoo_A;
        muda::DeviceDenseMatrix<StoreScalar>      debug_A;  // dense A for debug

        MatrixConverter<StoreScalar, 3> converter;

        bool empty_system = true;

        void compute_gradient(ComputeGradientInfo& info);
        void notify_line_search_result(const LineSearchFeedback& feedback);
        cudaStream_t stream() const noexcept { return cudaStreamLegacy; }

        bool        need_debug_dump = false;
        bool        need_solution_x_dump = false;
        std::string debug_dump_path;
    };

    SizeT dof_count() const;

    /**
     * @brief Interface to compute the gradient of the system.
     * 
     * The size of the gradient buffer should be equal to `dof_count()`.
     */
    void compute_gradient(ComputeGradientInfo& info);
    void notify_line_search_result(const LineSearchFeedback& feedback);
    cudaStream_t stream() const noexcept { return m_impl.stream(); }

    muda::LinearSystemContext& ctx() noexcept { return m_impl.ctx; }

    NewtonAssemblyMode newton_assembly_mode() const;

  protected:
    void do_build() override;

  private:
    friend class SimEngine;
    friend class LinearSolver;
    friend class DiagLinearSubsystem;
    friend class OffDiagLinearSubsystem;
    friend class LocalPreconditioner;
    friend class GlobalPreconditioner;
    friend class GlobalDiffSimManager;
    friend class CurrentFrameDiffDofReporter;

    void add_subsystem(DiagLinearSubsystem* subsystem);
    void add_subsystem(OffDiagLinearSubsystem* subsystem);
    void add_solver(LinearSolver* solver);
    void add_preconditioner(LocalPreconditioner* preconditioner);
    void add_preconditioner(GlobalPreconditioner* preconditioner);

    // only be called by SimEngine::do_init();
    void init();

    // only be called by SimEngine::do_advance()
    void solve();

    Impl m_impl;

    // local debug dump functions
    void _dump_A();
    void _dump_b();
    void _dump_x();
};
}  // namespace uipc::backend::cuda_mixed
