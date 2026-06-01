#include <catch2/catch_all.hpp>
#include <uipc/core/rcc_bonded_pt_oracle.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace
{
bool same_topo(const uipc::Vector4i& lhs, const uipc::Vector4i& rhs)
{
    return (lhs.array() == rhs.array()).all();
}
}  // namespace

TEST_CASE("rcc_bonded_pt_svts_rest_shape_offsets_and_orients",
          "[rcc_bonded_pt][oracle][rest_shape]")
{
    using namespace uipc;
    using namespace uipc::core;

    RCCBondedPTRestShapeInput input;
    input.topo = Vector4i{100, 200, 201, 202};
    input.point = Vector3{0.25, 0.25, 0.0};
    input.tri0 = Vector3{0.0, 0.0, 0.0};
    input.tri1 = Vector3{1.0, 0.0, 0.0};
    input.tri2 = Vector3{0.0, 1.0, 0.0};
    input.min_separate_distance = 0.05;

    auto rest = build_rcc_bonded_pt_rest_shape_svts(input);

    REQUIRE(rest.valid);
    CHECK(rest.signed_distance == Catch::Approx(0.0).margin(1e-14));
    CHECK(rest.conditioned_signed_distance == Catch::Approx(0.05).margin(1e-14));
    CHECK(rest.conditioned_point.z() == Catch::Approx(0.05).margin(1e-14));
    CHECK(rest.rest_volume == Catch::Approx(0.05 / 6.0).margin(1e-14));

    // The SVTS construction swaps tri0/tri1 when the initial Dm has negative det.
    CHECK(same_topo(rest.oriented_topo, Vector4i{100, 201, 200, 202}));

    Matrix3x3 I = rest.Dm * rest.Dm_inv;
    CHECK(I.isApprox(Matrix3x3::Identity(), 1e-12));
    CHECK(rest.Dm.determinant() == Catch::Approx(0.05).margin(1e-14));
}

TEST_CASE("rcc_bonded_pt_svts_rest_shape_rejects_degenerate_triangle",
          "[rcc_bonded_pt][oracle][rest_shape]")
{
    using namespace uipc;
    using namespace uipc::core;

    RCCBondedPTRestShapeInput input;
    input.topo = Vector4i{1, 2, 3, 4};
    input.point = Vector3{0.0, 0.0, 0.0};
    input.tri0 = Vector3{1.0, 0.0, 0.0};
    input.tri1 = Vector3{1.0, 0.0, 0.0};
    input.tri2 = Vector3{0.0, 1.0, 0.0};

    auto rest = build_rcc_bonded_pt_rest_shape_svts(input);

    CHECK_FALSE(rest.valid);
    CHECK(rest.rest_volume == Catch::Approx(0.0));
    CHECK(rest.Dm.isZero(0.0));
}

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

uipc::core::RCCBondedPTVirtualTetOracle eval_virtual_tet(
    const uipc::Vector12& q,
    const uipc::Matrix3x3& Dm_inv,
    uipc::Float rest_volume,
    bool project_hessian_to_spd)
{
    uipc::core::RCCBondedPTVirtualTetInput input;
    input.x0 = q.segment<3>(0);
    input.x1 = q.segment<3>(3);
    input.x2 = q.segment<3>(6);
    input.x3 = q.segment<3>(9);
    input.Dm_inv = Dm_inv;
    input.rest_volume = rest_volume;
    input.mu = 2.3;
    input.lambda = 5.1;
    input.dt = 0.25;
    input.project_hessian_to_spd = project_hessian_to_spd;
    return uipc::core::build_rcc_bonded_pt_virtual_tet_oracle(input);
}
}  // namespace

TEST_CASE("rcc_bonded_pt_virtual_tet_oracle_matches_finite_difference",
          "[rcc_bonded_pt][oracle][energy]")
{
    using namespace uipc;
    using namespace uipc::core;

    RCCBondedPTRestShapeInput rest_input;
    rest_input.topo = Vector4i{100, 200, 201, 202};
    rest_input.point = Vector3{0.25, 0.25, 0.0};
    rest_input.tri0 = Vector3{0.0, 0.0, 0.0};
    rest_input.tri1 = Vector3{1.0, 0.0, 0.0};
    rest_input.tri2 = Vector3{0.0, 1.0, 0.0};
    rest_input.min_separate_distance = 0.05;

    auto rest = build_rcc_bonded_pt_rest_shape_svts(rest_input);
    REQUIRE(rest.valid);

    Vector12 q_rest = pack_positions(rest.conditioned_point,
                                     rest_input.tri1,
                                     rest_input.tri0,
                                     rest_input.tri2);
    auto at_rest = eval_virtual_tet(q_rest, rest.Dm_inv, rest.rest_volume, true);
    REQUIRE(at_rest.valid);
    CHECK(at_rest.F.isApprox(Matrix3x3::Identity(), 1e-12));
    CHECK(at_rest.gradient.norm() == Catch::Approx(0.0).margin(1e-12));

    Vector12 q = q_rest;
    q[0] += 0.013;
    q[2] += 0.009;
    q[4] -= 0.011;
    q[6] += 0.017;
    q[10] += 0.015;

    auto oracle = eval_virtual_tet(q, rest.Dm_inv, rest.rest_volume, false);
    REQUIRE(oracle.valid);

    constexpr Float eps = 1e-6;
    Vector12 fd_gradient;
    Matrix12x12 fd_hessian;
    for(IndexT i = 0; i < 12; ++i)
    {
        Vector12 q_plus  = q;
        Vector12 q_minus = q;
        q_plus[i] += eps;
        q_minus[i] -= eps;

        auto plus  = eval_virtual_tet(q_plus, rest.Dm_inv, rest.rest_volume, false);
        auto minus = eval_virtual_tet(q_minus, rest.Dm_inv, rest.rest_volume, false);

        fd_gradient[i] = (plus.energy - minus.energy) / (2.0 * eps);
        fd_hessian.col(i) = (plus.gradient - minus.gradient) / (2.0 * eps);
    }

    CHECK((oracle.gradient - fd_gradient).cwiseAbs().maxCoeff() < 1e-7);
    CHECK((oracle.hessian - fd_hessian).cwiseAbs().maxCoeff() < 1e-5);

    auto projected = eval_virtual_tet(q, rest.Dm_inv, rest.rest_volume, true);
    REQUIRE(projected.valid);
    Eigen::SelfAdjointEigenSolver<Matrix12x12> solver(
        0.5 * (projected.hessian + projected.hessian.transpose()));
    CHECK(solver.eigenvalues().minCoeff() >= -1e-10);
}
