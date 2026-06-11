#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/elastic_moduli.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/constitution/rcc_adhesive.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/core/rcc_adhesion_state_accessor_feature.h>
#include <uipc/core/rcc_bonded_pt_state_accessor_feature.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>

namespace
{
using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

struct BetaStats
{
    SizeT count   = 0;
    Float mean    = std::numeric_limits<Float>::quiet_NaN();
    Float min     = std::numeric_limits<Float>::quiet_NaN();
    Float frac_09 = 0.0;
};

BetaStats beta_stats(const World& world)
{
    auto acc = world.features().find<RCCAdhesionStateAccessorFeature>();
    REQUIRE(acc != nullptr);

    vector<U64>   keys;
    vector<Float> betas;
    acc->dump_pt_state(keys, betas);

    BetaStats stats;
    stats.count = betas.size();
    if(betas.empty())
        return stats;

    stats.mean =
        std::accumulate(betas.begin(), betas.end(), Float{0}) / Float(betas.size());
    stats.min = *std::min_element(betas.begin(), betas.end());
    stats.frac_09 =
        Float(std::count_if(betas.begin(), betas.end(), [](Float b) { return b > 0.9; }))
        / Float(betas.size());
    return stats;
}

struct BondedStats
{
    SizeT               locked = 0;
    RCCBondedPTCounters counters;
};

BondedStats bonded_stats(const World& world)
{
    auto acc = world.features().find<RCCBondedPTStateAccessorFeature>();
    REQUIRE(acc != nullptr);

    BondedStats s;
    s.locked   = acc->locked_pair_count();
    s.counters = acc->counters();
    return s;
}

Float smooth_lerp(Float a, Float b, Float t)
{
    t       = std::clamp(t, Float{0}, Float{1});
    Float s = 0.5 - 0.5 * std::cos(std::numbers::pi_v<Float> * t);
    return a + (b - a) * s;
}

Matrix4x4 translate_y(Float y)
{
    Transform t = Transform::Identity();
    t.translate(Vector3{0.0, y, 0.0});
    return t.matrix();
}

Vector3 transform_point(const Matrix4x4& T, const Vector3& p)
{
    return T.template block<3, 3>(0, 0) * p + T.template block<3, 1>(0, 3);
}

void advance_to(World& world, SizeT frame)
{
    while(world.frame() < frame)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }
}

namespace cube_gate
{
constexpr int   PressStart       = 5;
constexpr int   ContactAt        = 60;
constexpr int   HoldUntil        = 100;
constexpr int   LiftUntil        = 200;
constexpr int   PrePullHoldUntil = 240;
constexpr int   PullUntil        = 360;
constexpr int   TotalFrames      = 400;

constexpr int   GridN          = 2;
constexpr Float CubeScale      = 0.3;
constexpr Float BottomInitialY = 0.17;
constexpr Float BottomPullY    = 0.17;
constexpr Float TopInitialY    = 0.70;
constexpr Float TopContactY    = 0.47;
constexpr Float TopLiftY       = 0.90;
constexpr Float TopPullY       = 1.20;
constexpr Float BottomLiftY    = BottomInitialY + (TopLiftY - TopContactY);

constexpr Float AdhesionCn  = 1.0e3;
constexpr Float AdhesionCt  = 1.0e3;
constexpr Float AdhesionW   = 3.8;
constexpr Float AdhesionEta = 0.2;

Float picker_y(int frame)
{
    if(frame < PressStart)
        return TopInitialY;
    if(frame < ContactAt)
        return smooth_lerp(TopInitialY,
                           TopContactY,
                           Float(frame - PressStart) / Float(ContactAt - PressStart));
    if(frame < HoldUntil)
        return TopContactY;
    if(frame < LiftUntil)
        return smooth_lerp(TopContactY,
                           TopLiftY,
                           Float(frame - HoldUntil) / Float(LiftUntil - HoldUntil));
    if(frame < PrePullHoldUntil)
        return TopLiftY;
    if(frame < PullUntil)
        return smooth_lerp(TopLiftY,
                           TopPullY,
                           Float(frame - PrePullHoldUntil)
                               / Float(PullUntil - PrePullHoldUntil));
    return TopPullY;
}

Float bottom_pull_y(int frame, bool adhesion_on)
{
    if(frame < LiftUntil)
        return BottomInitialY;

    Float start_y = adhesion_on ? BottomLiftY : BottomInitialY;
    if(frame < PrePullHoldUntil)
        return start_y;
    if(frame < PullUntil)
        return smooth_lerp(start_y,
                           BottomPullY,
                           Float(frame - PrePullHoldUntil)
                               / Float(PullUntil - PrePullHoldUntil));
    return BottomPullY;
}

SimplicialComplex make_subdivided_cube()
{
    vector<Vector3>  Vs;
    vector<Vector4i> Ts;

    auto vid = [](int i, int j, int k)
    { return (i * (GridN + 1) + j) * (GridN + 1) + k; };

    for(int i = 0; i <= GridN; ++i)
        for(int j = 0; j <= GridN; ++j)
            for(int k = 0; k <= GridN; ++k)
            {
                Float x = (Float(i) / GridN - 0.5) * CubeScale;
                Float y = (Float(j) / GridN - 0.5) * CubeScale;
                Float z = (Float(k) / GridN - 0.5) * CubeScale;
                Vs.push_back(Vector3{x, y, z});
            }

    const std::array<Vector3i, 8> corners = {Vector3i{0, 0, 0},
                                             Vector3i{1, 0, 0},
                                             Vector3i{0, 1, 0},
                                             Vector3i{1, 1, 0},
                                             Vector3i{0, 0, 1},
                                             Vector3i{1, 0, 1},
                                             Vector3i{0, 1, 1},
                                             Vector3i{1, 1, 1}};
    const std::array<Vector4i, 6> local_tets = {Vector4i{0, 1, 3, 7},
                                                Vector4i{0, 3, 2, 7},
                                                Vector4i{0, 2, 6, 7},
                                                Vector4i{0, 6, 4, 7},
                                                Vector4i{0, 4, 5, 7},
                                                Vector4i{0, 5, 1, 7}};

    for(int i = 0; i < GridN; ++i)
        for(int j = 0; j < GridN; ++j)
            for(int k = 0; k < GridN; ++k)
            {
                std::array<IndexT, 8> ids;
                for(SizeT q = 0; q < corners.size(); ++q)
                    ids[q] = vid(i + corners[q].x(),
                                 j + corners[q].y(),
                                 k + corners[q].z());

                for(auto tet : local_tets)
                {
                    Vector4i tet_ids{ids[tet[0]], ids[tet[1]], ids[tet[2]], ids[tet[3]]};
                    const Vector3& a = Vs[tet_ids[0]];
                    const Vector3& b = Vs[tet_ids[1]];
                    const Vector3& c = Vs[tet_ids[2]];
                    const Vector3& d = Vs[tet_ids[3]];
                    Float signed_volume = (b - a).dot((c - a).cross(d - a));
                    if(signed_volume < 0.0)
                        std::swap(tet_ids[2], tet_ids[3]);
                    Ts.push_back(tet_ids);
                }
            }

    auto cube = tetmesh(Vs, Ts);
    label_surface(cube);
    label_triangle_orient(cube);
    return cube;
}

struct HeightStats
{
    Float bottom_y = 0.0;
    Float top_y    = 0.0;
    Float gap_y    = 0.0;
};

HeightStats height_stats(const S<SimplicialComplexSlot>& slot)
{
    const auto& geo       = slot->geometry();
    auto        rest      = geo.positions().view();
    auto        transform = geo.transforms().view();

    Float bottom_sum = 0.0;
    Float top_sum    = 0.0;
    for(auto p : rest)
    {
        bottom_sum += transform_point(transform[0], p).y();
        top_sum += transform_point(transform[1], p).y();
    }

    HeightStats stats;
    stats.bottom_y = bottom_sum / Float(rest.size());
    stats.top_y    = top_sum / Float(rest.size());
    stats.gap_y    = stats.top_y - stats.bottom_y;
    return stats;
}

struct ContactGapStats
{
    SizeT bottom_count = 0;
    SizeT top_count    = 0;
    Float mean         = 0.0;
    Float min          = std::numeric_limits<Float>::infinity();
    Float max          = -std::numeric_limits<Float>::infinity();
};

ContactGapStats contact_gap_stats(const S<SimplicialComplexSlot>& slot)
{
    const auto& geo       = slot->geometry();
    auto        rest      = geo.positions().view();
    auto        transform = geo.transforms().view();

    Float rest_min_y = std::numeric_limits<Float>::infinity();
    Float rest_max_y = -std::numeric_limits<Float>::infinity();
    for(auto p : rest)
    {
        rest_min_y = std::min(rest_min_y, p.y());
        rest_max_y = std::max(rest_max_y, p.y());
    }

    vector<Float> bottom_top_y;
    vector<Float> top_bottom_y;
    constexpr Float eps = 1.0e-9;
    for(auto p : rest)
    {
        if(std::abs(p.y() - rest_max_y) <= eps)
            bottom_top_y.push_back(transform_point(transform[0], p).y());
        if(std::abs(p.y() - rest_min_y) <= eps)
            top_bottom_y.push_back(transform_point(transform[1], p).y());
    }

    ContactGapStats stats;
    stats.bottom_count = bottom_top_y.size();
    stats.top_count    = top_bottom_y.size();
    REQUIRE(bottom_top_y.size() == top_bottom_y.size());

    Float sum = 0.0;
    for(SizeT i = 0; i < bottom_top_y.size(); ++i)
    {
        Float gap = top_bottom_y[i] - bottom_top_y[i];
        sum += gap;
        stats.min = std::min(stats.min, gap);
        stats.max = std::max(stats.max, gap);
    }
    stats.mean = sum / Float(bottom_top_y.size());
    return stats;
}

S<SimplicialComplexSlot> build_scene(Scene& scene,
                                     bool   adhesion_on,
                                     bool   zero_adhesion_coeffs = false)
{
    AffineBodyConstitution  abd;
    SoftTransformConstraint stc;

    auto& tabular = scene.contact_tabular();
    tabular.default_model(0.5, 1.0e9);
    auto cube_contact   = tabular.default_element();
    auto ground_contact = tabular.create("ground");
    tabular.insert(cube_contact, ground_contact, 0.5, 1.0e9);

    if(adhesion_on)
    {
        RCCAdhesive adhesive;
        adhesive.default_model(tabular,
                               0.0,
                               0.0,
                               0.0,
                               1.0,
                               0.0,
                               0.0,
                               0.0,
                               false);
        if(zero_adhesion_coeffs)
        {
            // Distance-lock mode scene: RCCAdhesive applied (the lock driver
            // requires the tabular) with all adhesion coefficients zero —
            // bonding eligibility comes from adhesion_enabled alone; there is
            // no soft adhesion force at all.
            adhesive.set(tabular,
                         cube_contact,
                         cube_contact,
                         0.0,
                         0.0,
                         0.0,
                         1.0,
                         0.0,
                         0.0,
                         0.0,
                         true);
        }
        else
        {
            adhesive.set(tabular,
                         cube_contact,
                         cube_contact,
                         AdhesionCn,
                         AdhesionCt,
                         AdhesionW,
                         AdhesionEta,
                         1.0,
                         0.0,
                         1.0,
                         true);
        }
    }

    auto cube = make_subdivided_cube();
    cube.instances().resize(2);
    abd.apply_to(cube, 1.0e8);
    cube_contact.apply_to(cube);
    stc.apply_to(cube, Vector2{1.0e8, 0.0});

    auto transforms = view(cube.transforms());
    transforms[0]   = translate_y(BottomInitialY);
    transforms[1]   = translate_y(TopInitialY);

    auto cube_obj = scene.objects().create("subdivided_cubes");
    auto slots    = cube_obj->geometries().create(cube);

    auto ground_geo = ground(0.0);
    ground_contact.apply_to(ground_geo);
    auto ground_obj = scene.objects().create("ground");
    ground_obj->geometries().create(ground_geo);

    scene.animator().insert(
        *cube_obj,
        [adhesion_on](Animation::UpdateInfo& info)
        {
            auto geo =
                info.geo_slots()[0]->geometry().as<SimplicialComplex>();
            auto is_constrained =
                view(*geo->instances().find<IndexT>(builtin::is_constrained));
            auto aim_transform =
                view(*geo->instances().find<Matrix4x4>(builtin::aim_transform));

            int frame = info.frame() > 0 ? static_cast<int>(info.frame() - 1) : 0;
            aim_transform[1] = translate_y(picker_y(frame));
            is_constrained[1] = 1;

            if(frame < PrePullHoldUntil)
            {
                is_constrained[0] = 0;
                aim_transform[0]  = translate_y(BottomInitialY);
            }
            else
            {
                is_constrained[0] = 1;
                aim_transform[0]  = translate_y(bottom_pull_y(frame, adhesion_on));
            }
        });

    return slots.geometry;
}
}  // namespace cube_gate

namespace cloth_gate
{
constexpr int   PressStart       = 5;
constexpr int   ContactAt        = 60;
constexpr int   HoldUntil        = 100;
constexpr int   LiftUntil        = 200;
constexpr int   PrePullHoldUntil = 240;
constexpr int   PullUntil        = 360;
constexpr int   TotalFrames      = 400;

constexpr Float CubeScale     = 0.3;
constexpr int   ClothN        = 14;
constexpr Float ClothSize     = 0.44;
constexpr Float ClothSpacing  = ClothSize / ClothN;
constexpr Float ClothInitialY = 0.30;
constexpr int   PullVertexI   = ClothN / 2;
constexpr int   PullVertexJ   = ClothN / 2;

constexpr Float PressGap    = 0.010;
constexpr Float TopContactY = ClothInitialY + 0.5 * CubeScale + PressGap;
constexpr Float TopInitialY = TopContactY + 0.40;
constexpr Float TopLiftY    = TopContactY + 0.36;
constexpr Float TopPullY    = TopLiftY + 0.35;

constexpr Float ClothLiftY = ClothInitialY + (TopLiftY - TopContactY);
constexpr Float ClothPullY = ClothInitialY - 0.16;

constexpr Float AdhesionCn  = 3.0e-1;
constexpr Float AdhesionCt  = 3.0;
constexpr Float AdhesionW   = 0.5;
constexpr Float AdhesionEta = 0.5;

int vid(int i, int j)
{
    return i * (ClothN + 1) + j;
}

Float cube_y(int frame)
{
    if(frame < PressStart)
        return TopInitialY;
    if(frame < ContactAt)
        return smooth_lerp(TopInitialY,
                           TopContactY,
                           Float(frame - PressStart) / Float(ContactAt - PressStart));
    if(frame < HoldUntil)
        return TopContactY;
    if(frame < LiftUntil)
        return smooth_lerp(TopContactY,
                           TopLiftY,
                           Float(frame - HoldUntil) / Float(LiftUntil - HoldUntil));
    if(frame < PrePullHoldUntil)
        return TopLiftY;
    if(frame < PullUntil)
        return smooth_lerp(TopLiftY,
                           TopPullY,
                           Float(frame - PrePullHoldUntil)
                               / Float(PullUntil - PrePullHoldUntil));
    return TopPullY;
}

Float cloth_pull_y(int frame, bool adhesion_on)
{
    if(frame < LiftUntil)
        return ClothInitialY;

    Float start_y = adhesion_on ? ClothLiftY : ClothInitialY;
    if(frame < PrePullHoldUntil)
        return start_y;
    if(frame < PullUntil)
        return smooth_lerp(start_y,
                           ClothPullY,
                           Float(frame - PrePullHoldUntil)
                               / Float(PullUntil - PrePullHoldUntil));
    return ClothPullY;
}

SimplicialComplex make_cloth_mesh()
{
    vector<Vector3>  Vs;
    vector<Vector3i> Fs;
    for(int i = 0; i <= ClothN; ++i)
        for(int j = 0; j <= ClothN; ++j)
            Vs.push_back(Vector3{i * ClothSpacing - 0.5 * ClothSize,
                                 ClothInitialY,
                                 j * ClothSpacing - 0.5 * ClothSize});

    for(int i = 0; i < ClothN; ++i)
        for(int j = 0; j < ClothN; ++j)
        {
            int v00 = vid(i, j);
            int v10 = vid(i + 1, j);
            int v01 = vid(i, j + 1);
            int v11 = vid(i + 1, j + 1);
            Fs.push_back(Vector3i{v00, v01, v11});
            Fs.push_back(Vector3i{v00, v11, v10});
        }

    auto cloth = trimesh(Vs, Fs);
    label_surface(cloth);
    mesh_partition(cloth, 16);
    return cloth;
}

struct Slots
{
    S<SimplicialComplexSlot> cube;
    S<SimplicialComplexSlot> cloth;
};

struct HeightStats
{
    Float cube_y  = 0.0;
    Float cloth_y = 0.0;
    Float gap_y   = 0.0;
};

HeightStats height_stats(const Slots& slots)
{
    const auto& cube_geo    = slots.cube->geometry();
    auto        cube_pos    = cube_geo.positions().view();
    auto        cube_T      = cube_geo.transforms().view()[0];
    const auto& cloth_geo   = slots.cloth->geometry();
    auto        cloth_pos   = cloth_geo.positions().view();

    Float cube_sum = 0.0;
    for(auto p : cube_pos)
        cube_sum += transform_point(cube_T, p).y();

    Float cloth_sum = 0.0;
    for(auto p : cloth_pos)
        cloth_sum += p.y();

    HeightStats stats;
    stats.cube_y  = cube_sum / Float(cube_pos.size());
    stats.cloth_y = cloth_sum / Float(cloth_pos.size());
    stats.gap_y   = stats.cube_y - stats.cloth_y;
    return stats;
}

struct BottomContactStats
{
    SizeT count = 0;
    Float mean  = 0.0;
    Float min   = std::numeric_limits<Float>::infinity();
    Float max   = -std::numeric_limits<Float>::infinity();
};

BottomContactStats bottom_contact_stats(const Slots& slots)
{
    const auto& cube_geo  = slots.cube->geometry();
    auto        cube_pos  = cube_geo.positions().view();
    auto        cube_T    = cube_geo.transforms().view()[0];
    const auto& cloth_geo = slots.cloth->geometry();
    auto        cloth_pos = cloth_geo.positions().view();

    Vector3 cube_min = Vector3::Constant(std::numeric_limits<Float>::infinity());
    Vector3 cube_max = Vector3::Constant(-std::numeric_limits<Float>::infinity());
    for(auto p : cube_pos)
    {
        Vector3 q = transform_point(cube_T, p);
        cube_min  = cube_min.cwiseMin(q);
        cube_max  = cube_max.cwiseMax(q);
    }

    BottomContactStats stats;
    Float              sum = 0.0;
    for(auto p : cloth_pos)
    {
        if(p.x() >= cube_min.x() && p.x() <= cube_max.x() && p.z() >= cube_min.z()
           && p.z() <= cube_max.z())
        {
            Float gap = cube_min.y() - p.y();
            ++stats.count;
            sum       += gap;
            stats.min = std::min(stats.min, gap);
            stats.max = std::max(stats.max, gap);
        }
    }

    REQUIRE(stats.count > 0);
    stats.mean = sum / Float(stats.count);
    return stats;
}

Slots build_scene(Scene& scene, bool adhesion_on)
{
    AffineBodyConstitution  abd;
    NeoHookeanShell         nhs;
    SoftPositionConstraint  spc;
    SoftTransformConstraint stc;

    auto& tabular = scene.contact_tabular();
    tabular.default_model(0.5, 1.0e9);
    auto cube_contact  = tabular.default_element();
    auto cloth_contact = tabular.create("cloth");
    tabular.insert(cloth_contact, cloth_contact, 0.0, 1.0e9);
    tabular.insert(cloth_contact, cube_contact, 0.5, 1.0e9);

    if(adhesion_on)
    {
        RCCAdhesive adhesive;
        adhesive.default_model(tabular,
                               0.0,
                               0.0,
                               0.0,
                               1.0,
                               0.0,
                               0.0,
                               0.0,
                               false);
        adhesive.set(tabular,
                     cloth_contact,
                     cube_contact,
                     AdhesionCn,
                     AdhesionCt,
                     AdhesionW,
                     AdhesionEta,
                     1.0,
                     0.0,
                     1.0,
                     true);
        adhesive.set(tabular,
                     cloth_contact,
                     cloth_contact,
                     0.0,
                     0.0,
                     0.0,
                     1.0,
                     0.0,
                     0.0,
                     0.0,
                     false);
    }

    Transform pre = Transform::Identity();
    pre.scale(CubeScale);
    SimplicialComplexIO io{pre};

    auto cube = io.read(fmt::format("{}cube.msh", AssetDir::tetmesh_path()));
    label_surface(cube);
    label_triangle_orient(cube);
    abd.apply_to(cube, 1.0e8);
    cube_contact.apply_to(cube);
    stc.apply_to(cube, Vector2{1.0e8, 0.0});
    view(cube.transforms())[0] = translate_y(TopInitialY);

    auto cube_obj   = scene.objects().create("press_cube");
    auto cube_slots = cube_obj->geometries().create(cube);

    auto cloth = make_cloth_mesh();
    nhs.apply_to(cloth, ElasticModuli2D::youngs_poisson(1.0e7, 0.4));
    cloth_contact.apply_to(cloth);
    spc.apply_to(cloth, 2.0e5);
    if(adhesion_on)
        RCCAdhesive::set_sticky_side(cloth, +1);

    vector<Vector3> rest_positions;
    {
        auto pos = cloth.positions().view();
        rest_positions.assign(pos.begin(), pos.end());
    }
    const int pull_vertex = vid(PullVertexI, PullVertexJ);

    auto cloth_obj   = scene.objects().create("cloth");
    auto cloth_slots = cloth_obj->geometries().create(cloth);

    scene.animator().insert(
        *cube_obj,
        [](Animation::UpdateInfo& info)
        {
            auto geo =
                info.geo_slots()[0]->geometry().as<SimplicialComplex>();
            auto is_constrained =
                view(*geo->instances().find<IndexT>(builtin::is_constrained));
            auto aim_transform =
                view(*geo->instances().find<Matrix4x4>(builtin::aim_transform));
            int frame = info.frame() > 0 ? static_cast<int>(info.frame() - 1) : 0;
            is_constrained[0] = 1;
            aim_transform[0]  = translate_y(cube_y(frame));
        });

    scene.animator().insert(
        *cloth_obj,
        [rest_positions = std::move(rest_positions), pull_vertex, adhesion_on](
            Animation::UpdateInfo& info)
        {
            auto geo =
                info.geo_slots()[0]->geometry().as<SimplicialComplex>();
            auto is_constrained =
                view(*geo->vertices().find<IndexT>(builtin::is_constrained));
            auto aim_position =
                view(*geo->vertices().find<Vector3>(builtin::aim_position));
            int frame = info.frame() > 0 ? static_cast<int>(info.frame() - 1) : 0;

            std::fill(is_constrained.begin(), is_constrained.end(), IndexT{0});
            if(frame < PrePullHoldUntil)
                return;

            const auto& rest          = rest_positions[pull_vertex];
            is_constrained[pull_vertex] = 1;
            aim_position[pull_vertex] =
                Vector3{rest.x(), cloth_pull_y(frame, adhesion_on), rest.z()};
        });

    return Slots{cube_slots.geometry, cloth_slots.geometry};
}
}  // namespace cloth_gate
}  // namespace

TEST_CASE("rcc_adhesion_subdivided_cube_lift_hold_release_gate",
          "[rcc_adhesion][gate][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;

    logger::set_level(spdlog::level::err);

    auto out_path = fmt::format("{}cube/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));
    Engine engine{"cuda", out_path};
    World  world{engine};

    auto config                             = test::Scene::default_config();
    config["dt"]                           = 0.01;
    config["gravity"]                      = Vector3{0.0, -9.8, 0.0};
    config["contact"]["enable"]            = true;
    config["contact"]["friction"]["enable"] = true;
    config["contact"]["d_hat"]             = 0.02;
    config["linear_system"]["tol_rate"]    = 1.0e-3;
    config["extras"]["strict_mode"]["enable"] = false;
    test::Scene::dump_config(config, out_path);

    Scene scene{config};
    auto  cube_slot = cube_gate::build_scene(scene, true);

    world.init(scene);
    REQUIRE(world.is_valid());

    advance_to(world, cube_gate::PrePullHoldUntil);
    auto hold_height  = cube_gate::height_stats(cube_slot);
    auto hold_contact = cube_gate::contact_gap_stats(cube_slot);
    auto hold_beta    = beta_stats(world);

    CAPTURE(hold_height.bottom_y,
            hold_height.gap_y,
            hold_contact.bottom_count,
            hold_contact.top_count,
            hold_contact.min,
            hold_contact.max,
            hold_beta.count,
            hold_beta.min,
            hold_beta.frac_09);

    REQUIRE(hold_height.bottom_y > cube_gate::BottomLiftY - 0.04);
    REQUIRE(hold_height.gap_y < 0.35);
    REQUIRE(hold_contact.bottom_count == SizeT((cube_gate::GridN + 1) * (cube_gate::GridN + 1)));
    REQUIRE(hold_contact.top_count == SizeT((cube_gate::GridN + 1) * (cube_gate::GridN + 1)));
    REQUIRE(hold_contact.min >= 0.0);
    REQUIRE(hold_contact.max <= 0.03);
    REQUIRE(hold_beta.count >= 8);
    REQUIRE(hold_beta.min >= 0.95);
    REQUIRE(hold_beta.frac_09 == Catch::Approx(1.0));

    advance_to(world, cube_gate::PullUntil);
    auto pull_height  = cube_gate::height_stats(cube_slot);
    auto pull_contact = cube_gate::contact_gap_stats(cube_slot);
    auto pull_beta    = beta_stats(world);

    CAPTURE(pull_height.bottom_y,
            pull_height.gap_y,
            pull_contact.min,
            pull_beta.count,
            pull_beta.min,
            pull_beta.frac_09);

    REQUIRE(pull_height.bottom_y < cube_gate::BottomInitialY + 0.02);
    REQUIRE(pull_height.gap_y > 0.8);
    REQUIRE(pull_contact.min > 0.6);
    REQUIRE(pull_beta.frac_09 == Catch::Approx(0.0));

    advance_to(world, cube_gate::TotalFrames);
}

TEST_CASE("rcc_adhesion_cube_cloth_lift_hold_release_gate",
          "[rcc_adhesion][gate][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;

    logger::set_level(spdlog::level::err);

    auto out_path = fmt::format("{}cloth/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));
    Engine engine{"cuda", out_path};
    World  world{engine};

    auto config                                = test::Scene::default_config();
    config["dt"]                              = 0.01;
    config["gravity"]                         = Vector3{0.0, 0.0, 0.0};
    config["contact"]["enable"]               = true;
    config["contact"]["friction"]["enable"]   = true;
    config["contact"]["d_hat"]                = 0.02;
    config["linear_system"]["tol_rate"]       = 1.0e-3;
    config["extras"]["strict_mode"]["enable"] = false;
    test::Scene::dump_config(config, out_path);

    Scene scene{config};
    auto  slots = cloth_gate::build_scene(scene, true);

    world.init(scene);
    REQUIRE(world.is_valid());

    advance_to(world, cloth_gate::PrePullHoldUntil);
    auto hold_height = cloth_gate::height_stats(slots);
    auto hold_bottom = cloth_gate::bottom_contact_stats(slots);
    auto hold_beta   = beta_stats(world);

    CAPTURE(hold_height.cloth_y,
            hold_height.gap_y,
            hold_bottom.count,
            hold_bottom.mean,
            hold_bottom.min,
            hold_bottom.max,
            hold_beta.count,
            hold_beta.min,
            hold_beta.frac_09);

    REQUIRE(hold_height.cloth_y > cloth_gate::ClothLiftY - 0.04);
    REQUIRE(hold_height.gap_y < 0.20);
    REQUIRE(hold_bottom.count >= 80);
    REQUIRE(std::abs(hold_bottom.mean) <= 0.03);
    REQUIRE(hold_bottom.min >= -0.06);
    REQUIRE(hold_bottom.max <= 0.06);
    REQUIRE(hold_beta.count >= 80);
    REQUIRE(hold_beta.min >= 0.99);
    REQUIRE(hold_beta.frac_09 == Catch::Approx(1.0));

    advance_to(world, cloth_gate::PullUntil);
    auto pull_height = cloth_gate::height_stats(slots);
    auto pull_bottom = cloth_gate::bottom_contact_stats(slots);
    auto pull_beta   = beta_stats(world);

    CAPTURE(pull_height.cloth_y,
            pull_height.gap_y,
            pull_bottom.count,
            pull_bottom.mean,
            pull_bottom.min,
            pull_bottom.max,
            pull_beta.count,
            pull_beta.min,
            pull_beta.frac_09);

    REQUIRE(pull_height.cloth_y < cloth_gate::ClothInitialY);
    REQUIRE(pull_height.gap_y > 0.8);
    REQUIRE(pull_bottom.mean > 0.5);
    REQUIRE(pull_beta.frac_09 == Catch::Approx(0.0));

    advance_to(world, cloth_gate::TotalFrames);
}

// pt_lift_release no-penetration gate: enable bonded-PT acceleration AND the
// pre-CCD skip, so locked PT pairs are removed from CCD broadphase and the ABD
// virtual-tet energy is the ONLY thing preventing penetration. The gate proves
// (1) bonded locks form, (2) no visible penetration during press/hold/lift while
// CCD is skipped, and (3) the lower cube is still lifted through the bonded
// energy. See docs/architecture.md "CCD Removal Precondition".
TEST_CASE("rcc_bonded_pt_cube_lift_hold_no_penetration_gate",
          "[rcc_bonded_pt][scene][pt_lift_release][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;

    logger::set_level(spdlog::level::err);

    auto out_path = fmt::format("{}bonded_cube/",
                                AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));
    Engine engine{"cuda", out_path};
    World  world{engine};

    auto config                               = test::Scene::default_config();
    config["dt"]                              = 0.01;
    config["gravity"]                         = Vector3{0.0, -9.8, 0.0};
    config["contact"]["enable"]               = true;
    config["contact"]["friction"]["enable"]   = true;
    config["contact"]["d_hat"]                = 0.02;
    config["linear_system"]["tol_rate"]       = 1.0e-3;
    config["extras"]["strict_mode"]["enable"] = false;
    // Bonded-PT acceleration with the pre-CCD skip enabled.
    config["rcc_bonded_pt_enabled"]             = 1;
    config["rcc_bonded_pt_skip_ccd"]            = 1;
    config["rcc_bonded_pt_beta_lock_threshold"] = 0.9;
    config["rcc_bonded_pt_energy_model"]        = "abd_ortho";
    config["rcc_bonded_pt_kappa"]               = 1.0e8;  // production gate value
    test::Scene::dump_config(config, out_path);

    Scene scene{config};
    auto  cube_slot = cube_gate::build_scene(scene, true);

    world.init(scene);
    REQUIRE(world.is_valid());

    // PenTol < d_hat (0.02): a larger interpenetration would be "visible".
    constexpr Float PenTol = 0.01;

    SizeT max_locked      = 0;
    Float min_gap         = std::numeric_limits<Float>::infinity();
    Float worst_pen_frame = -1.0;

    // Press -> contact -> hold -> lift. Sample the contact-face gap every frame
    // (host-side) and the bonded locked count periodically (device download).
    for(int f = cube_gate::ContactAt; f <= cube_gate::PrePullHoldUntil; ++f)
    {
        advance_to(world, SizeT(f));
        auto cg = cube_gate::contact_gap_stats(cube_slot);
        if(cg.min < min_gap)
        {
            min_gap         = cg.min;
            worst_pen_frame = Float(f);
        }
        if(f % 20 == 0)
            max_locked = std::max(max_locked, bonded_stats(world).locked);
    }

    auto end_bonded  = bonded_stats(world);
    max_locked       = std::max(max_locked, end_bonded.locked);
    auto hold_height = cube_gate::height_stats(cube_slot);

    CAPTURE(max_locked,
            end_bonded.counters.candidate_count,
            end_bonded.counters.locked_count,
            end_bonded.counters.filter_skipped_count,
            end_bonded.counters.duplicate_suppressed_count,
            end_bonded.counters.degenerate_rejected_count,
            min_gap,
            worst_pen_frame,
            hold_height.bottom_y,
            hold_height.gap_y);

    // (1) Bonded locks actually formed: the pre-CCD path engaged.
    REQUIRE(max_locked >= 1);
    // (2) No visible penetration while CCD is skipped for locked pairs.
    REQUIRE(min_gap >= -PenTol);
    // (3) Adhesion still lifts the lower cube through the bonded energy.
    REQUIRE(hold_height.bottom_y > cube_gate::BottomLiftY - 0.06);

    advance_to(world, cube_gate::TotalFrames);
}

// Phase 7 distance-lock scene gate: bonded virtual tets with NO adhesion
// energy at all. RCCAdhesive is applied with Cn = Ct = 0 (the lock driver
// requires the tabular; adhesion_enabled = 1 grants bond eligibility), and
// the lock gate is the distance band d < xi + c*d_hat evaluated at end of
// step. The gate proves (1) bonds form by distance alone, (2) no visible
// penetration while CCD is skipped, (3) the lower cube is carried purely by
// the bonded energy (no soft adhesion force exists in this mode), (4) locked
// entries carry the sentinel beta 1.0 through the accessor dump, (5) the
// unchanged release gates fire on the forced pull and full separation leaves
// zero locks, and (6) the distance-rejection counter is live.
TEST_CASE("rcc_bonded_pt_distance_lock_cube_lift_release_gate",
          "[rcc_bonded_pt][scene][distance_lock][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;

    logger::set_level(spdlog::level::err);

    auto out_path = fmt::format("{}distance_lock_cube/",
                                AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));
    Engine engine{"cuda", out_path};
    World  world{engine};

    auto config                               = test::Scene::default_config();
    config["dt"]                              = 0.01;
    config["gravity"]                         = Vector3{0.0, -9.8, 0.0};
    config["contact"]["enable"]               = true;
    config["contact"]["friction"]["enable"]   = true;
    config["contact"]["d_hat"]                = 0.02;
    config["linear_system"]["tol_rate"]       = 1.0e-3;
    config["extras"]["strict_mode"]["enable"] = false;
    config["rcc_bonded_pt_enabled"]           = 1;
    config["rcc_bonded_pt_skip_ccd"]          = 1;
    config["rcc_bonded_pt_energy_model"]      = "abd_ortho";
    config["rcc_bonded_pt_kappa"]             = 1.0e8;
    // Distance-lock mode: lock when d < xi + c*d_hat (xi = 0 here). The press
    // equilibrium gap of this fixture is ~0.0178 (the IPC barrier at
    // d_hat = 0.02 carries the press load well before contact), so the band
    // must cover it: c = 0.95 -> band 0.019. Keeping c < 1 leaves an active
    // DCD window (0.019, 0.02) that the approach transits, which keeps the
    // distance-rejection counter assertion meaningful.
    config["rcc_bonded_pt_distance_lock"]       = 1;
    config["rcc_bonded_pt_distance_lock_ratio"] = 0.95;
    // Distance mode locks on first contact; only face-interior feet may bond
    // (the default 0 would mass-lock edge/corner sliver tets).
    config["rcc_bonded_pt_lock_face_interior_only"] = 1;
    // Deliberately a beta-mode value > 1: the distance-lock driver must clamp
    // the global threshold to <= 1 so it cannot veto indicator locks.
    config["rcc_bonded_pt_beta_lock_threshold"] = 1.5;
    // Gap release band wider than the lock band (release_gap > c*d_hat), per
    // the anti-churn guidance; the forced pull separates far beyond it.
    config["rcc_bonded_pt_release_gap"] = 0.05;
    test::Scene::dump_config(config, out_path);

    Scene scene{config};
    auto  cube_slot =
        cube_gate::build_scene(scene, true, /*zero_adhesion_coeffs=*/true);

    world.init(scene);
    REQUIRE(world.is_valid());

    constexpr Float PenTol = 0.01;

    SizeT max_locked      = 0;
    Float min_gap         = std::numeric_limits<Float>::infinity();
    Float worst_pen_frame = -1.0;

    for(int f = cube_gate::ContactAt; f <= cube_gate::PrePullHoldUntil; ++f)
    {
        advance_to(world, SizeT(f));
        auto cg = cube_gate::contact_gap_stats(cube_slot);
        if(cg.min < min_gap)
        {
            min_gap         = cg.min;
            worst_pen_frame = Float(f);
        }
        if(f % 20 == 0)
            max_locked = std::max(max_locked, bonded_stats(world).locked);
    }

    auto pre_pull   = bonded_stats(world);
    max_locked      = std::max(max_locked, pre_pull.locked);
    auto hold_height = cube_gate::height_stats(cube_slot);
    auto bstats      = beta_stats(world);

    CAPTURE(max_locked,
            pre_pull.counters.candidate_count,
            pre_pull.counters.locked_count,
            pre_pull.counters.released_count,
            pre_pull.counters.distance_rejected_count,
            pre_pull.counters.policy_rejected_count,
            pre_pull.counters.degenerate_rejected_count,
            pre_pull.counters.duplicate_suppressed_count,
            min_gap,
            worst_pen_frame,
            hold_height.bottom_y,
            bstats.count,
            bstats.min);

    // (1) Bonds formed by the distance band alone (beta does not exist).
    REQUIRE(max_locked >= 1);
    // (2) No visible penetration while CCD is skipped for locked pairs.
    REQUIRE(min_gap >= -PenTol);
    // (3) The lower cube is carried purely by the bonded virtual tets: with
    // Cn = Ct = 0 there is no soft adhesion force that could lift it.
    REQUIRE(hold_height.bottom_y > cube_gate::BottomLiftY - 0.06);
    // (4) Sentinel-beta contract: the adhesion state dump contains exactly
    // the bonded locked pairs (no soft beta state exists), all at beta 1.0.
    REQUIRE(bstats.count == pre_pull.locked);
    if(bstats.count > 0)
        REQUIRE(bstats.min == 1.0);
    // (6) The distance-band rejection counter is live: the gradual press
    // transits the DCD-active-but-outside-band window (c*d_hat, d_hat).
    REQUIRE(pre_pull.counters.distance_rejected_count >= 1);

    // (5) Forced pull: the unchanged release gates fire (gap growth exceeds
    // rcc_bonded_pt_release_gap) and full separation leaves zero locks.
    advance_to(world, cube_gate::PullUntil);
    auto post_pull = bonded_stats(world);
    CAPTURE(post_pull.locked, post_pull.counters.released_count);
    REQUIRE(post_pull.counters.released_count > pre_pull.counters.released_count);

    advance_to(world, cube_gate::TotalFrames);
    auto end_bonded = bonded_stats(world);
    CAPTURE(end_bonded.locked);
    REQUIRE(end_bonded.locked == 0);
}
