#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <contact_system/rcc_bonded_pt_virtual_tet_reporter.h>
#include <muda/buffer/device_buffer.h>
#include <uipc/core/rcc_bonded_pt_oracle.h>

namespace
{
uipc::Vector12 pack_positions(const uipc::Vector3& x0,
                              const uipc::Vector3& x1,
                              const uipc::Vector3& x2,
                              const uipc::Vector3& x3)
{
    uipc::Vector12 q;
    q.segment<3>(0) = x0;
    q.segment<3>(3) = x1;
    q.segment<3>(6) = x2;
    q.segment<3>(9) = x3;
    return q;
}
}  // namespace

TEST_CASE("rcc_bonded_pt_virtual_tet_reporter_matches_cpu_oracle",
          "[rcc_bonded_pt][reporter][oracle][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    RCCBondedPTRestShapeInput rest_input;
    rest_input.topo = Vector4i{0, 1, 2, 3};
    rest_input.point = Vector3{0.25, 0.25, 0.0};
    rest_input.tri0 = Vector3{0.0, 0.0, 0.0};
    rest_input.tri1 = Vector3{1.0, 0.0, 0.0};
    rest_input.tri2 = Vector3{0.0, 1.0, 0.0};
    rest_input.min_separate_distance = 0.05;
    rest_input.triangle_degeneracy_tol = 1e-12;

    const auto rest = build_rcc_bonded_pt_rest_shape_svts(rest_input);
    REQUIRE(rest.valid);

    std::vector<Vector3> h_positions(4);
    h_positions[0] = rest.conditioned_point + Vector3{0.013, 0.0, 0.009};
    h_positions[1] = rest_input.tri0 + Vector3{0.0, -0.011, 0.0};
    h_positions[2] = rest_input.tri1 + Vector3{0.017, 0.0, 0.0};
    h_positions[3] = rest_input.tri2 + Vector3{0.0, 0.015, 0.0};

    const Vector12 q = pack_positions(h_positions[0],
                                      h_positions[rest.oriented_topo[1]],
                                      h_positions[rest.oriented_topo[2]],
                                      h_positions[rest.oriented_topo[3]]);

    constexpr Float mu = 2.3;
    constexpr Float lambda = 5.1;
    constexpr Float dt = 0.25;

    RCCBondedPTVirtualTetInput oracle_input;
    oracle_input.x0 = q.segment<3>(0);
    oracle_input.x1 = q.segment<3>(3);
    oracle_input.x2 = q.segment<3>(6);
    oracle_input.x3 = q.segment<3>(9);
    oracle_input.Dm_inv = rest.Dm_inv;
    oracle_input.rest_volume = rest.rest_volume;
    oracle_input.mu = mu;
    oracle_input.lambda = lambda;
    oracle_input.dt = dt;
    oracle_input.project_hessian_to_spd = true;
    const auto oracle = build_rcc_bonded_pt_virtual_tet_oracle(oracle_input);
    REQUIRE(oracle.valid);

    std::vector<Vector4i> h_topos = {rest.oriented_topo};
    std::vector<Matrix3x3> h_dm_inv = {rest.Dm_inv};
    std::vector<Float> h_rest_volume = {rest.rest_volume};

    DeviceBuffer<Vector4i> d_topos;
    DeviceBuffer<Matrix3x3> d_dm_inv;
    DeviceBuffer<Float> d_rest_volume;
    DeviceBuffer<Vector3> d_positions;
    DeviceBuffer<Float> d_energies;
    DeviceBuffer<Vector12> d_gradients;
    DeviceBuffer<Matrix12x12> d_hessians;

    d_topos.copy_from(h_topos);
    d_dm_inv.copy_from(h_dm_inv);
    d_rest_volume.copy_from(h_rest_volume);
    d_positions.copy_from(h_positions);
    d_energies.resize(1);
    d_gradients.resize(1);
    d_hessians.resize(1);

    RCCBondedPTVirtualTetReporter::Impl impl;
    impl.set_material(mu, lambda);
    impl.compute_dense_energy_gradient_hessian(d_topos.view(),
                                               d_dm_inv.view(),
                                               d_rest_volume.view(),
                                               d_positions.view(),
                                               dt,
                                               d_energies.view(),
                                               d_gradients.view(),
                                               d_hessians.view());

    std::vector<Float> h_energies(1);
    std::vector<Vector12> h_gradients(1);
    std::vector<Matrix12x12> h_hessians(1);
    d_energies.view().copy_to(h_energies.data());
    d_gradients.view().copy_to(h_gradients.data());
    d_hessians.view().copy_to(h_hessians.data());

    CHECK(h_energies[0] == Catch::Approx(oracle.energy).epsilon(1e-8).margin(1e-10));
    CHECK((h_gradients[0] - oracle.gradient).cwiseAbs().maxCoeff()
          == Catch::Approx(0.0).margin(1e-8));
    CHECK((h_hessians[0] - oracle.hessian).cwiseAbs().maxCoeff()
          == Catch::Approx(0.0).margin(1e-6));
}
