#pragma once
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <utils/structured_contact_assembly_sink.h>

namespace uipc::backend::cuda_mixed
{
class SimplexNormalContact : public ContactReporter
{
  public:
    using ContactReporter::ContactReporter;
    using StoreScalar = ContactReporter::StoreScalar;
    using EnergyScalar = ContactReporter::EnergyScalar;
    constexpr static SizeT PTHalfHessianSize = 4 * (4 + 1) / 2;  // 4 vertices, symmetric matrix
    constexpr static SizeT EEHalfHessianSize = 4 * (4 + 1) / 2;  // 4 vertices, symmetric matrix
    constexpr static SizeT PEHalfHessianSize = 3 * (3 + 1) / 2;  // 3 vertices, symmetric matrix
    constexpr static SizeT PPHalfHessianSize = 2 * (2 + 1) / 2;  // 2 vertices, symmetric matrix

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        muda::CBuffer2DView<ContactCoeff> contact_tabular() const;
        muda::CBufferView<Vector4i>       PTs() const;
        muda::CBufferView<Vector4i>       EEs() const;
        muda::CBufferView<Vector3i>       PEs() const;
        muda::CBufferView<Vector2i>       PPs() const;

        muda::CBufferView<Float>   thicknesses() const;
        muda::CBufferView<Vector3> positions() const;
        muda::CBufferView<Vector3> prev_positions() const;
        muda::CBufferView<Vector3> rest_positions() const;
        muda::CBufferView<IndexT>  contact_element_ids() const;
        Float                      d_hat() const;
        muda::CBufferView<Float>   d_hats() const;
        Float                      dt() const;
        Float                      eps_velocity() const;

      private:
        friend class SimplexNormalContact;
        Impl* m_impl;
    };

    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }
        auto PT_gradients() const noexcept { return m_PT_gradients; }
        auto PT_hessians() const noexcept { return m_PT_hessians; }
        auto EE_gradients() const noexcept { return m_EE_gradients; }
        auto EE_hessians() const noexcept { return m_EE_hessians; }
        auto PE_gradients() const noexcept { return m_PE_gradients; }
        auto PE_hessians() const noexcept { return m_PE_hessians; }
        auto PP_gradients() const noexcept { return m_PP_gradients; }
        auto PP_hessians() const noexcept { return m_PP_hessians; }
        bool gradient_only() const noexcept { return m_gradient_only; }
        bool hessian_only() const noexcept { return m_hessian_only; }
        bool structured_hessian() const noexcept { return m_structured_hessian; }
        auto structured_hessian_sink() const noexcept { return m_structured_sink; }

      private:
        friend class SimplexNormalContact;
        muda::DoubletVectorView<StoreScalar, 3> m_PT_gradients;
        muda::TripletMatrixView<StoreScalar, 3> m_PT_hessians;

        muda::DoubletVectorView<StoreScalar, 3> m_EE_gradients;
        muda::TripletMatrixView<StoreScalar, 3> m_EE_hessians;

        muda::DoubletVectorView<StoreScalar, 3> m_PE_gradients;
        muda::TripletMatrixView<StoreScalar, 3> m_PE_hessians;

        muda::DoubletVectorView<StoreScalar, 3> m_PP_gradients;
        muda::TripletMatrixView<StoreScalar, 3> m_PP_hessians;
        bool                              m_gradient_only = false;
        bool                              m_hessian_only = false;
        bool                              m_structured_hessian = false;
        StructuredContactAssemblySink<StoreScalar, ActivePolicy::SolveScalar> m_structured_sink;
    };

    class BuildInfo
    {
      public:
    };

    class EnergyInfo : public BaseInfo
    {
      public:
        EnergyInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        muda::BufferView<EnergyScalar> PT_energies() const noexcept
        {
            return m_PT_energies;
        }
        muda::BufferView<EnergyScalar> EE_energies() const noexcept
        {
            return m_EE_energies;
        }
        muda::BufferView<EnergyScalar> PE_energies() const noexcept
        {
            return m_PE_energies;
        }
        muda::BufferView<EnergyScalar> PP_energies() const noexcept
        {
            return m_PP_energies;
        }

      private:
        friend class SimplexNormalContact;
        muda::BufferView<EnergyScalar> m_PT_energies;
        muda::BufferView<EnergyScalar> m_EE_energies;
        muda::BufferView<EnergyScalar> m_PE_energies;
        muda::BufferView<EnergyScalar> m_PP_energies;
    };

    class Impl
    {
      public:
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;
        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;

        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;

        // constraint count

        SizeT PT_count = 0;
        SizeT EE_count = 0;
        SizeT PE_count = 0;
        SizeT PP_count = 0;

        Float                   dt = 0;
        muda::DeviceVar<IndexT> selected_count;

        Float reserve_ratio = 1.1;

        template <typename T>
        void loose_resize(muda::DeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }

        muda::CBufferView<EnergyScalar>    PT_energies;
        muda::CDoubletVectorView<StoreScalar, 3> PT_gradients;
        muda::CTripletMatrixView<StoreScalar, 3> PT_hessians;

        muda::CBufferView<EnergyScalar>    EE_energies;
        muda::CDoubletVectorView<StoreScalar, 3> EE_gradients;
        muda::CTripletMatrixView<StoreScalar, 3> EE_hessians;

        muda::CBufferView<EnergyScalar>    PE_energies;
        muda::CDoubletVectorView<StoreScalar, 3> PE_gradients;
        muda::CTripletMatrixView<StoreScalar, 3> PE_hessians;

        muda::CBufferView<EnergyScalar>    PP_energies;
        muda::CDoubletVectorView<StoreScalar, 3> PP_gradients;
        muda::CTripletMatrixView<StoreScalar, 3> PP_hessians;
    };

    muda::CBufferView<Vector4i>        PTs() const;
    muda::CBufferView<EnergyScalar>    PT_energies() const;
    muda::CDoubletVectorView<StoreScalar, 3> PT_gradients() const;
    muda::CTripletMatrixView<StoreScalar, 3> PT_hessians() const;

    muda::CBufferView<Vector4i>        EEs() const;
    muda::CBufferView<EnergyScalar>    EE_energies() const;
    muda::CDoubletVectorView<StoreScalar, 3> EE_gradients() const;
    muda::CTripletMatrixView<StoreScalar, 3> EE_hessians() const;

    muda::CBufferView<Vector3i>        PEs() const;
    muda::CBufferView<EnergyScalar>    PE_energies() const;
    muda::CDoubletVectorView<StoreScalar, 3> PE_gradients() const;
    muda::CTripletMatrixView<StoreScalar, 3> PE_hessians() const;

    muda::CBufferView<Vector2i>        PPs() const;
    muda::CBufferView<EnergyScalar>    PP_energies() const;
    muda::CDoubletVectorView<StoreScalar, 3> PP_gradients() const;
    muda::CTripletMatrixView<StoreScalar, 3> PP_hessians() const;

  protected:
    virtual void do_build(BuildInfo& info)           = 0;
    virtual void do_compute_energy(EnergyInfo& info) = 0;
    virtual void do_assemble(ContactInfo& info)      = 0;

  private:
    virtual void do_report_energy_extent(GlobalContactManager::EnergyExtentInfo& info) override final;
    virtual void do_compute_energy(GlobalContactManager::EnergyInfo& info) override final;
    virtual void do_report_gradient_hessian_extent(
        GlobalContactManager::GradientHessianExtentInfo& info) override final;
    virtual void do_assemble(GlobalContactManager::GradientHessianInfo& info) override final;
    virtual bool do_supports_structured_hessian() const override final;
    virtual void do_assemble_structured_hessian(
        GlobalDyTopoEffectManager::StructuredHessianInfo& info) override final;
    virtual void do_build(ContactReporter::BuildInfo& info) override final;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda_mixed
