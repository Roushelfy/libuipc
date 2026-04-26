#pragma once
#include <finite_element/finite_element_constitution.h>
#include <utils/assembly_sink.h>

namespace uipc::backend::cuda_mixed
{
class Codim2DConstitution : public FiniteElementConstitution
{
  public:
    using FiniteElementConstitution::FiniteElementConstitution;
    using StoreScalar = FiniteElementElastics::StoreScalar;
    using EnergyScalar = FiniteElementElastics::EnergyScalar;

    class BuildInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(Codim2DConstitution* impl, SizeT index_in_dim, Float dt)
            : m_impl(impl)
            , m_index_in_dim(index_in_dim)
            , m_dt(dt)
        {
        }

        muda::CBufferView<Vector3>  xs() const noexcept;
        muda::CBufferView<Vector3>  x_bars() const noexcept;
        muda::CBufferView<Float>    rest_areas() const noexcept;
        muda::CBufferView<Float>    thicknesses() const noexcept;
        muda::CBufferView<Vector3i> indices() const noexcept;
        const FiniteElementMethod::ConstitutionInfo& constitution_info() const noexcept;
        Float dt() const noexcept;

      protected:
        SizeT                m_index_in_dim = ~0ull;
        Codim2DConstitution* m_impl         = nullptr;
        Float                m_dt           = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Codim2DConstitution*    impl,
                          SizeT                   index_in_dim,
                          Float                   dt,
                          muda::BufferView<EnergyScalar> energies)
            : BaseInfo(impl, index_in_dim, dt)
            , m_energies(energies)
        {
        }

        auto energies() const noexcept { return m_energies; }

      private:
        muda::BufferView<EnergyScalar> m_energies;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Codim2DConstitution* impl,
                                   SizeT                index_in_dim,
                                   bool                 gradient_only,
                                   Float                dt,
                                   muda::DoubletVectorView<StoreScalar, 3> gradients,
                                   muda::TripletMatrixView<StoreScalar, 3> hessians,
                                   StructuredDeviceAssemblySink<
                                       StoreScalar,
                                       GlobalLinearSystem::SolveScalar> structured_sink = {},
                                   IndexT old_dof_offset = 0,
                                   muda::CBufferView<IndexT> fixed_vertices = {},
                                   bool identity_fixed_diagonal = false,
                                   bool write_gradients = true)
            : BaseInfo(impl, index_in_dim, dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_structured_sink(structured_sink)
            , m_old_dof_offset(old_dof_offset)
            , m_fixed_vertices(fixed_vertices)
            , m_identity_fixed_diagonal(identity_fixed_diagonal)
            , m_write_gradients(write_gradients)
            , m_gradient_only(gradient_only)
        {
        }

        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept
        {
            return structured_assembly() ? muda::TripletMatrixView<StoreScalar, 3>{}
                                         : m_hessians;
        }
        auto gradient_only() const noexcept { return m_gradient_only; }
        bool structured_assembly() const noexcept
        {
            return m_structured_sink.valid();
        }
        auto sink() const noexcept
        {
            return LocalAssemblySink<StoreScalar, GlobalLinearSystem::SolveScalar, 3>{
                m_gradients,
                hessians(),
                m_gradient_only,
                m_structured_sink,
                m_old_dof_offset,
                m_fixed_vertices,
                m_identity_fixed_diagonal,
                m_write_gradients};
        }

      private:
        muda::DoubletVectorView<StoreScalar, 3> m_gradients;
        muda::TripletMatrixView<StoreScalar, 3> m_hessians;
        StructuredDeviceAssemblySink<StoreScalar, GlobalLinearSystem::SolveScalar> m_structured_sink;
        IndexT                            m_old_dof_offset = 0;
        muda::CBufferView<IndexT>         m_fixed_vertices;
        bool                              m_identity_fixed_diagonal = false;
        bool                              m_write_gradients = true;
        bool                              m_gradient_only = false;
    };

  protected:
    virtual void do_build(BuildInfo& info)                  = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;
    const FiniteElementMethod::ConstitutionInfo& constitution_info() const noexcept;

  private:
    friend class FiniteElementMethod;
    virtual void do_build(FiniteElementConstitution::BuildInfo& info) override final;
    virtual void do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info) override final;
    virtual void do_compute_gradient_hessian(
        FiniteElementConstitution::ComputeGradientHessianInfo& info) override final;
    virtual IndexT get_dim() const noexcept override final;
};
}  // namespace uipc::backend::cuda_mixed
