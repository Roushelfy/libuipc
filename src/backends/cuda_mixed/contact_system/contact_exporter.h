#pragma once
#include <sim_system.h>
#include <muda/buffer/buffer_view.h>
#include <uipc/geometry/attribute_slot.h>
#include <vector>

namespace uipc::backend::cuda_mixed
{
class ContactExporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    std::string_view prim_type() const noexcept;

    class BuildInfo
    {
      public:
    };

  protected:
    template <typename Energy>
    static void copy_contact_energies_to_geometry(muda::CBufferView<Energy> energies,
                                                  geometry::Geometry&        energy_geo)
    {
        auto energy = energy_geo.instances().find<Float>("energy");
        if(!energy)
        {
            energy = energy_geo.instances().create<Float>("energy", Float{0.0});
        }

        auto energy_view = view(*energy);
        std::vector<Energy> host_energies(energies.size());
        energies.copy_to(host_energies.data());

        UIPC_ASSERT(host_energies.size() == energy_view.size(),
                    "Energy buffer size mismatch, expected {}, got {}",
                    energy_view.size(),
                    host_energies.size());

        for(SizeT i = 0; i < host_energies.size(); ++i)
        {
            energy_view[i] = static_cast<Float>(host_energies[i]);
        }
    }

    virtual void             do_build(BuildInfo& info)      = 0;
    virtual std::string_view get_prim_type() const noexcept = 0;

    virtual void get_contact_energy(std::string_view    prim_type,
                                    geometry::Geometry& energy_geo) = 0;

    virtual void get_contact_gradient(std::string_view    prim_type,
                                      geometry::Geometry& vert_grad) = 0;

    virtual void get_contact_hessian(std::string_view    prim_type,
                                     geometry::Geometry& vert_hess) = 0;

  private:
    virtual void do_build() override final;

    friend class ContactExporterManager;
    void contact_energy(std::string_view prim_type, geometry::Geometry& energy_geo);
    void contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad);
    void contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess);
};
}  // namespace uipc::backend::cuda_mixed
