#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <contact_system/rcc_bonded_pt_virtual_tet_reporter.h>
#include <muda/buffer/device_buffer.h>
#include <uipc/core/rcc_bonded_pt_oracle.h>

#include <algorithm>
#include <cmath>

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

uipc::Float max_relative_error(const uipc::Vector12& actual,
                               const uipc::Vector12& expected)
{
    uipc::Float max_error = 0.0;
    for(uipc::IndexT i = 0; i < actual.size(); ++i)
    {
        const uipc::Float denom =
            std::max<uipc::Float>(1.0, std::abs(expected[i]));
        max_error = std::max(max_error,
                             std::abs(actual[i] - expected[i]) / denom);
    }
    return max_error;
}

uipc::Float max_relative_error(const uipc::Matrix12x12& actual,
                               const uipc::Matrix12x12& expected)
{
    uipc::Float max_error = 0.0;
    for(uipc::IndexT i = 0; i < actual.rows(); ++i)
    {
        for(uipc::IndexT j = 0; j < actual.cols(); ++j)
        {
            const uipc::Float denom =
                std::max<uipc::Float>(1.0, std::abs(expected(i, j)));
            max_error = std::max(max_error,
                                 std::abs(actual(i, j) - expected(i, j))
                                     / denom);
        }
    }
    return max_error;
}
}  // namespace

TEST_CASE("rcc_bonded_pt_abd_virtual_tet_reporter_matches_cpu_oracle",
          "[rcc_bonded_pt][reporter][abd_oracle][cuda]")
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

    constexpr Float kappa = 1e8;
    constexpr Float dt = 0.25;

    RCCBondedPTVirtualTetInput oracle_input;
    oracle_input.x0 = q.segment<3>(0);
    oracle_input.x1 = q.segment<3>(3);
    oracle_input.x2 = q.segment<3>(6);
    oracle_input.x3 = q.segment<3>(9);
    oracle_input.Dm_inv = rest.Dm_inv;
    oracle_input.rest_volume = rest.rest_volume;
    oracle_input.energy_model = RCCBondedPTVirtualTetEnergyModel::ABDOrtho;
    oracle_input.kappa = kappa;
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
    impl.set_material(kappa);
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
    CHECK(max_relative_error(h_gradients[0], oracle.gradient) < 1e-10);
    CHECK(max_relative_error(h_hessians[0], oracle.hessian) < 1e-10);
}

namespace
{
// Shared deformed virtual-tet configuration for the NeoHookean tests.
struct NeoHookeanCase
{
    uipc::core::RCCBondedPTRestShape rest;
    std::vector<uipc::Vector3>       positions;  // topo order
    uipc::Vector12                   q;          // oriented order (x0..x3)
    uipc::Float                      mu, lambda, dt;
};

NeoHookeanCase make_neohookean_case()
{
    using namespace uipc;
    using namespace uipc::core;
    NeoHookeanCase c;

    RCCBondedPTRestShapeInput rest_input;
    rest_input.topo                  = Vector4i{0, 1, 2, 3};
    rest_input.point                 = Vector3{0.25, 0.25, 0.0};
    rest_input.tri0                  = Vector3{0.0, 0.0, 0.0};
    rest_input.tri1                  = Vector3{1.0, 0.0, 0.0};
    rest_input.tri2                  = Vector3{0.0, 1.0, 0.0};
    rest_input.min_separate_distance = 0.05;
    rest_input.triangle_degeneracy_tol = 1e-12;
    c.rest = build_rcc_bonded_pt_rest_shape_svts(rest_input);

    c.positions.resize(4);
    c.positions[0] = c.rest.conditioned_point + Vector3{0.013, 0.0, 0.009};
    c.positions[1] = rest_input.tri0 + Vector3{0.0, -0.011, 0.0};
    c.positions[2] = rest_input.tri1 + Vector3{0.017, 0.0, 0.0};
    c.positions[3] = rest_input.tri2 + Vector3{0.0, 0.015, 0.0};

    c.q.segment<3>(0) = c.positions[0];
    c.q.segment<3>(3) = c.positions[c.rest.oriented_topo[1]];
    c.q.segment<3>(6) = c.positions[c.rest.oriented_topo[2]];
    c.q.segment<3>(9) = c.positions[c.rest.oriented_topo[3]];

    // NeoHookean Lamé from a representative tape material (E=5e7, nu=0.45).
    const Float E = 5e7, nu = 0.45;
    c.mu     = E / (2.0 * (1.0 + nu));
    c.lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    c.dt     = 0.25;
    return c;
}
}  // namespace

TEST_CASE("rcc_bonded_pt_neohookean_virtual_tet_reporter_matches_cpu_oracle",
          "[rcc_bonded_pt][reporter][neohookean][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const NeoHookeanCase c = make_neohookean_case();
    REQUIRE(c.rest.valid);

    RCCBondedPTVirtualTetInput oracle_input;
    oracle_input.x0           = c.q.segment<3>(0);
    oracle_input.x1           = c.q.segment<3>(3);
    oracle_input.x2           = c.q.segment<3>(6);
    oracle_input.x3           = c.q.segment<3>(9);
    oracle_input.Dm_inv       = c.rest.Dm_inv;
    oracle_input.rest_volume  = c.rest.rest_volume;
    oracle_input.energy_model = RCCBondedPTVirtualTetEnergyModel::StableNeoHookean;
    oracle_input.mu           = c.mu;
    oracle_input.lambda       = c.lambda;
    oracle_input.dt           = c.dt;
    oracle_input.project_hessian_to_spd = true;
    const auto oracle = build_rcc_bonded_pt_virtual_tet_oracle(oracle_input);
    REQUIRE(oracle.valid);

    std::vector<Vector4i>  h_topos       = {c.rest.oriented_topo};
    std::vector<Matrix3x3> h_dm_inv      = {c.rest.Dm_inv};
    std::vector<Float>     h_rest_volume = {c.rest.rest_volume};

    DeviceBuffer<Vector4i>    d_topos;
    DeviceBuffer<Matrix3x3>   d_dm_inv;
    DeviceBuffer<Float>       d_rest_volume;
    DeviceBuffer<Vector3>     d_positions;
    DeviceBuffer<Float>       d_energies;
    DeviceBuffer<Vector12>    d_gradients;
    DeviceBuffer<Matrix12x12> d_hessians;
    d_topos.copy_from(h_topos);
    d_dm_inv.copy_from(h_dm_inv);
    d_rest_volume.copy_from(h_rest_volume);
    d_positions.copy_from(c.positions);
    d_energies.resize(1);
    d_gradients.resize(1);
    d_hessians.resize(1);

    RCCBondedPTVirtualTetReporter::Impl impl;
    impl.set_material_neohookean(c.mu, c.lambda);
    impl.compute_dense_energy_gradient_hessian(d_topos.view(),
                                               d_dm_inv.view(),
                                               d_rest_volume.view(),
                                               d_positions.view(),
                                               c.dt,
                                               d_energies.view(),
                                               d_gradients.view(),
                                               d_hessians.view());

    std::vector<Float>        h_energies(1);
    std::vector<Vector12>     h_gradients(1);
    std::vector<Matrix12x12>  h_hessians(1);
    d_energies.view().copy_to(h_energies.data());
    d_gradients.view().copy_to(h_gradients.data());
    d_hessians.view().copy_to(h_hessians.data());

    // GPU reporter (sym::stable_neo_hookean_3d) vs CPU oracle (independent
    // reimplementation of the same closed form) — agreement to ~machine eps.
    CHECK(h_energies[0] == Catch::Approx(oracle.energy).epsilon(1e-8).margin(1e-10));
    CHECK(max_relative_error(h_gradients[0], oracle.gradient) < 1e-10);
    CHECK(max_relative_error(h_hessians[0], oracle.hessian) < 1e-10);
}

TEST_CASE("rcc_bonded_pt_neohookean_oracle_matches_finite_difference",
          "[rcc_bonded_pt][oracle][neohookean][fd]")
{
    using namespace uipc;
    using namespace uipc::core;

    const NeoHookeanCase c = make_neohookean_case();
    REQUIRE(c.rest.valid);

    // Unprojected Hessian so finite differences of the gradient match it
    // directly (the SPD projection is identity only near rest).
    RCCBondedPTVirtualTetInput base;
    base.Dm_inv               = c.rest.Dm_inv;
    base.rest_volume          = c.rest.rest_volume;
    base.energy_model         = RCCBondedPTVirtualTetEnergyModel::StableNeoHookean;
    base.mu                   = c.mu;
    base.lambda               = c.lambda;
    base.dt                   = c.dt;
    base.project_hessian_to_spd = false;

    auto set_X = [&](RCCBondedPTVirtualTetInput& in, const Vector12& X)
    {
        in.x0 = X.segment<3>(0);
        in.x1 = X.segment<3>(3);
        in.x2 = X.segment<3>(6);
        in.x3 = X.segment<3>(9);
    };
    auto energy_of = [&](const Vector12& X)
    {
        RCCBondedPTVirtualTetInput in = base;
        set_X(in, X);
        return build_rcc_bonded_pt_virtual_tet_oracle(in).energy;
    };
    auto grad_of = [&](const Vector12& X)
    {
        RCCBondedPTVirtualTetInput in = base;
        set_X(in, X);
        return build_rcc_bonded_pt_virtual_tet_oracle(in).gradient;
    };

    Vector12 X0 = c.q;
    set_X(base, X0);
    const auto ref = build_rcc_bonded_pt_virtual_tet_oracle(base);
    REQUIRE(ref.valid);

    const Float eps = 1e-6;

    // grad[k] ?= d E / d X_k  (central difference)
    Vector12 g_fd = Vector12::Zero();
    for(IndexT k = 0; k < 12; ++k)
    {
        Vector12 xp = X0, xm = X0;
        xp[k] += eps;
        xm[k] -= eps;
        g_fd[k] = (energy_of(xp) - energy_of(xm)) / (2.0 * eps);
    }
    CHECK(max_relative_error(g_fd, ref.gradient) < 1e-5);

    // hessian.col(k) ?= d grad / d X_k  (central difference)
    Matrix12x12 H_fd = Matrix12x12::Zero();
    for(IndexT k = 0; k < 12; ++k)
    {
        Vector12 xp = X0, xm = X0;
        xp[k] += eps;
        xm[k] -= eps;
        H_fd.col(k) = (grad_of(xp) - grad_of(xm)) / (2.0 * eps);
    }
    CHECK(max_relative_error(H_fd, ref.hessian) < 1e-4);
}
