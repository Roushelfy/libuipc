#pragma once
#include <sim_system.h>
#include <energy_component_flags.h>
#include <global_geometry/global_vertex_manager.h>
#include <muda/ext/linear_system.h>
#include <utils/offset_count_collection.h>
#include <algorithm/matrix_converter.h>
#include <dytopo_effect_system/dytopo_classify_info.h>
#include <linear_system/assembly_mode.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/socu_native_descriptors.h>
#include <linear_system/socu_contact_topology_stamp.h>
#include <mixed_precision/policy.h>
#include <utils/structured_contact_assembly_sink.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda_mixed
{
class DyTopoEffectReporter;
class DyTopoEffectReceiver;
class ABDLinearSubsystem;
class FEMLinearSubsystem;
class AffineBodyDynamics;
class FiniteElementMethod;
class AffineBodyVertexReporter;
class FiniteElementVertexReporter;
struct SocuContactAssemblyPlan;
struct SocuContactAssemblyPlanM2Workspace;

class GlobalDyTopoEffectManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    using StoreScalar = ActivePolicy::StoreScalar;
    using EnergyScalar = ActivePolicy::EnergyScalar;

    class Impl;

    class GradientHessianExtentInfo
    {
      public:
        bool gradient_only() const { return m_gradient_only; }
        void gradient_count(SizeT count) noexcept { m_gradient_count = count; }
        void hessian_count(SizeT count) noexcept { m_hessian_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectReporter;

        bool  m_gradient_only  = false;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
    };

    class GradientHessianInfo
    {
      public:
        bool gradient_only() const noexcept { return m_gradient_only; }
        muda::DoubletVectorView<StoreScalar, 3> gradients() const noexcept
        {
            return m_gradients;
        }
        muda::TripletMatrixView<StoreScalar, 3> hessians() const noexcept
        {
            return m_hessians;
        }


      private:
        friend class Impl;
        bool                              m_gradient_only = false;
        muda::DoubletVectorView<StoreScalar, 3> m_gradients;
        muda::TripletMatrixView<StoreScalar, 3> m_hessians;
    };

    class EnergyExtentInfo
    {
      public:
        void energy_count(SizeT count) noexcept { m_energy_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectLineSearchReporter;
        SizeT m_energy_count = 0;
    };

    class EnergyInfo
    {
      public:
        muda::BufferView<EnergyScalar> energies() const { return m_energies; }
        bool                           is_initial() const { return m_is_initial; }

      private:
        friend class DyTopoEffectLineSearchReporter;
        muda::BufferView<EnergyScalar> m_energies;
        bool                           m_is_initial = false;
    };

    using ClassifyInfo = DyTopoClassifyInfo;

    class ClassifiedDyTopoEffectInfo
    {
      public:
        muda::CDoubletVectorView<StoreScalar, 3> gradients() const noexcept
        {
            return m_gradients;
        }
        muda::CTripletMatrixView<StoreScalar, 3> hessians() const noexcept
        {
            return m_hessians;
        }

      private:
        friend class Impl;
        muda::CDoubletVectorView<StoreScalar, 3> m_gradients;
        muda::CTripletMatrixView<StoreScalar, 3> m_hessians;
    };

    class StructuredHessianInfo
    {
      public:
        using ContactSink =
            StructuredContactAssemblySink<StoreScalar, ActivePolicy::SolveScalar>;

        ContactSink contact_sink() const noexcept { return m_contact_sink; }
        cudaStream_t stream() const noexcept { return m_stream; }
        muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors() const noexcept
        {
            return m_vertex_descriptors;
        }
        IndexT descriptor_epoch() const noexcept { return m_descriptor_epoch; }

      private:
        friend class Impl;
        ContactSink  m_contact_sink;
        cudaStream_t m_stream = cudaStreamLegacy;
        muda::CBufferView<SocuNativeVertexDescriptor> m_vertex_descriptors;
        IndexT m_descriptor_epoch = 0;
    };

    class ComputeDyTopoEffectInfo
    {
      public:
        void gradient_only(bool v) noexcept { m_gradient_only = v; }
        void component_flags(EnergyComponentFlags v) noexcept
        {
            m_component_flags = v;
        }
        void assembly_mode(NewtonAssemblyMode v) noexcept { m_assembly_mode = v; }

      private:
        friend class Impl;
        bool                 m_gradient_only   = false;
        EnergyComponentFlags m_component_flags = EnergyComponentFlags::All;
        NewtonAssemblyMode   m_assembly_mode   = NewtonAssemblyMode::FullSparse;
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);
        void _assemble(ComputeDyTopoEffectInfo& info);
        void _convert_matrix(ComputeDyTopoEffectInfo& info);
        void _distribute(ComputeDyTopoEffectInfo& info);
        void assemble_structured_hessian(
            GlobalLinearSystem::StructuredAssemblyInfo& info);
        void ensure_structured_vertex_descriptors(
            GlobalLinearSystem::StructuredAssemblyInfo& info);
        void build_socu_contact_assembly_plan_m2(
            SocuContactAssemblyPlan&            plan,
            SocuContactAssemblyPlanM2Workspace& workspace,
            const SocuVertexSidePlanKey&        side_key,
            const SocuContactProgramPlanKey&    program_key,
            muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
            StructuredContactOffbandPolicy offband_policy,
            SocuVertexSideCoverageMode     coverage_mode,
            bool                            build_hot_block_plan,
            SizeT                           hot_block_threshold,
            cudaStream_t                   stream);
        SizeT contact_set_signature();
        SocuContactTopologyStamp contact_topology_stamp(cudaStream_t stream);

        struct StructuredVertexDescriptorCacheKey
        {
            // The descriptor table is an ordering-epoch cache. Current FEM fixed
            // flags, ABD body fixed flags, and ABD vertex-to-body mappings are
            // treated as structural data inside one descriptor epoch. If any of
            // those buffers become mutable in-place, the mutating system must
            // bump descriptor_epoch or this key must grow a content signature.
            IndexT epoch = 0;
            SizeT  global_vertex_count = 0;
            SizeT  horizon = 0;
            SizeT  block_size = 0;
            const void* old_to_chain_data = nullptr;
            SizeT       old_to_chain_size = 0;

            IndexT fem_vertex_offset = -1;
            IndexT fem_vertex_count = 0;
            IndexT fem_old_dof_offset = -1;
            const void* fem_fixed_data = nullptr;
            SizeT       fem_fixed_size = 0;

            IndexT abd_vertex_offset = -1;
            IndexT abd_vertex_count = 0;
            IndexT abd_old_dof_offset = -1;
            IndexT abd_body_count = 0;
            const void* abd_vertex_to_body_data = nullptr;
            SizeT       abd_vertex_to_body_size = 0;
            const void* abd_body_is_fixed_data = nullptr;
            SizeT       abd_body_is_fixed_size = 0;

            bool operator==(const StructuredVertexDescriptorCacheKey&) const noexcept =
                default;
        };

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;
        SimSystemSlot<ABDLinearSubsystem> abd_linear_subsystem;
        SimSystemSlot<FEMLinearSubsystem> fem_linear_subsystem;
        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
        SimSystemSlot<FiniteElementMethod> finite_element_method;
        SimSystemSlot<AffineBodyVertexReporter> affine_body_vertex_reporter;
        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

        Float reserve_ratio = 1.1;


        /***********************************************************************
        *                              Reporter                                *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReporter> dytopo_effect_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> contact_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> non_contact_reporters;

        OffsetCountCollection<IndexT> reporter_energy_offsets_counts;
        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        muda::DeviceTripletMatrix<StoreScalar, 3> collected_dytopo_effect_hessian;
        muda::DeviceDoubletVector<StoreScalar, 3> collected_dytopo_effect_gradient;

        MatrixConverter<StoreScalar, 3>        matrix_converter;
        muda::DeviceBCOOMatrix<StoreScalar, 3> sorted_dytopo_effect_hessian;
        muda::DeviceBCOOVector<StoreScalar, 3> sorted_dytopo_effect_gradient;

        /***********************************************************************
        *                               Receiver                               *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReceiver> dytopo_effect_receivers;

        muda::DeviceVar<Vector2i>  gradient_range;
        muda::DeviceBuffer<IndexT> selected_hessian;
        muda::DeviceBuffer<IndexT> selected_hessian_offsets;

        vector<muda::DeviceTripletMatrix<StoreScalar, 3>> classified_dytopo_effect_hessians;
        vector<muda::DeviceDoubletVector<StoreScalar, 3>> classified_dytopo_effect_gradients;
        muda::DeviceBuffer<SocuNativeVertexDescriptor> structured_vertex_descriptors;
        StructuredVertexDescriptorCacheKey structured_vertex_descriptor_key;
        IndexT structured_vertex_descriptor_epoch = 0;
        SocuContactTopologyHashWorkspace contact_topology_hash_workspace;
        SocuContactTopologyStampCache    contact_topology_stamp_cache;

        void loose_resize_entries(muda::DeviceTripletMatrix<StoreScalar, 3>& m, SizeT size);
        void loose_resize_entries(muda::DeviceDoubletVector<StoreScalar, 3>& v, SizeT size);
        template <typename T>
        void loose_resize(muda::DeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }
    };

    muda::CBCOOVectorView<StoreScalar, 3> gradients() const noexcept;
    muda::CBCOOMatrixView<StoreScalar, 3> hessians() const noexcept;

    void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);
    void assemble_structured_hessian(GlobalLinearSystem::StructuredAssemblyInfo& info);
    void ensure_structured_vertex_descriptors(
        GlobalLinearSystem::StructuredAssemblyInfo& info);
    void build_socu_contact_assembly_plan_m2(
        SocuContactAssemblyPlan&            plan,
        SocuContactAssemblyPlanM2Workspace& workspace,
        const SocuVertexSidePlanKey&        side_key,
        const SocuContactProgramPlanKey&    program_key,
        muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors,
        StructuredContactOffbandPolicy offband_policy,
        SocuVertexSideCoverageMode     coverage_mode,
        bool                            build_hot_block_plan,
        SizeT                           hot_block_threshold,
        cudaStream_t                   stream);
    SizeT contact_set_signature();
    SocuContactTopologyStamp contact_topology_stamp(cudaStream_t stream);

  protected:
    virtual void do_build() override;

  private:
    friend class DyTopoEffectLineSearchReporter;
    void init();

    friend class SimEngine;
    // only be called by SimEngine
    void compute_dytopo_effect();

    friend class DyTopoEffectReporter;
    void add_reporter(DyTopoEffectReporter* reporter);
    friend class DyTopoEffectReceiver;
    void add_receiver(DyTopoEffectReceiver* receiver);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda_mixed
