#include <catch2/catch_all.hpp>
#include <uipc/core/rcc_bonded_pt_oracle.h>

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
