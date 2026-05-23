#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/constitution/rcc_adhesive.h>

// Smoke test for RCC adhesion v1 (Cn/Ct only, β = initial_beta, no evolution).
// Two ABD cubes stacked; the bottom is fixed. With strong adhesion, the top
// cube should stay bonded to the bottom. The test passes if the simulation
// advances through all frames without producing NaN positions.
TEST_CASE("rcc_adhesion_smoke", "[rcc_adhesion]")
{
    using namespace uipc;
    using namespace uipc::geometry;
    using namespace uipc::core;
    using namespace uipc::constitution;

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    std::string out_path =
        fmt::format("{}rcc/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));

    Engine engine{"cuda", out_path};
    World  world{engine};

    auto config       = test::Scene::default_config();
    config["gravity"] = Vector3{0, 0, 0};
    config["contact"]["friction"]["enable"] = true;

    test::Scene::dump_config(config, out_path);
    Scene scene{config};

    {
        AffineBodyConstitution abd;

        auto& tabular         = scene.contact_tabular();
        auto  default_element = tabular.default_element();
        tabular.default_model(0.5, 1.0_GPa);

        RCCAdhesive adhesive;
        adhesive.default_model(tabular,
                               /*Cn=*/1e6,
                               /*Ct=*/1e6,
                               /*W=*/1.0,
                               /*eta=*/2.0,
                               /*bonding_rate=*/1.0,
                               /*p0=*/0.0,
                               /*initial_beta=*/1.0,
                               /*enabled=*/true);

        Transform pre = Transform::Identity();
        pre.scale(0.3);
        SimplicialComplexIO io{pre};

        auto              object = scene.objects().create("cubes");
        SimplicialComplex cube   = io.read(fmt::format("{}cube.msh", tetmesh_dir));
        label_surface(cube);
        label_triangle_orient(cube);

        cube.instances().resize(2);
        abd.apply_to(cube, 100.0_MPa);
        default_element.apply_to(cube);

        auto trans_view    = view(cube.transforms());
        auto is_fixed      = cube.instances().find<IndexT>(builtin::is_fixed);
        auto is_fixed_view = view(*is_fixed);

        Transform t0     = Transform::Identity();
        t0.translation() = Vector3::Zero();
        trans_view[0]    = t0.matrix();
        is_fixed_view[0] = 1;

        Transform t1     = Transform::Identity();
        t1.translation() = Vector3::UnitY() * 0.305;
        trans_view[1]    = t1.matrix();
        is_fixed_view[1] = 0;

        object->geometries().create(cube);
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", out_path, 0));

    constexpr int max_frames = 30;
    while(world.frame() < max_frames)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", out_path, world.frame()));
    }
}

// Demo: animate the top cube to press the bottom cube, then lift up.
// With RCC adhesion ENABLED, the bottom cube should stick to the top one and
// follow it up. With adhesion DISABLED (or β=0), the bottom cube falls under
// gravity once the top retracts. Run both SECTIONs and compare the OBJ sequences:
//   output/tests/sim_case/rcc_adhesion_smoke.cpp/lift_with_adhesion/scene_surface*.obj
//   output/tests/sim_case/rcc_adhesion_smoke.cpp/lift_without_adhesion/scene_surface*.obj
TEST_CASE("rcc_adhesion_pick_and_lift", "[rcc_adhesion][demo]")
{
    using namespace uipc;
    using namespace uipc::geometry;
    using namespace uipc::core;
    using namespace uipc::constitution;

    bool        adhesion_on = false;
    std::string label;

    SECTION("lift_with_adhesion")
    {
        adhesion_on = true;
        label       = "lift_with_adhesion";
    }
    SECTION("lift_without_adhesion")
    {
        adhesion_on = false;
        label       = "lift_without_adhesion";
    }

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    std::string out_path = fmt::format(
        "{}{}/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE), label);

    Engine engine{"cuda", out_path};
    World  world{engine};

    auto  config = test::Scene::default_config();
    Float dt     = 0.01;
    config["dt"] = dt;
    // Gravity points down; the ground half-plane holds the pickee in place.
    // Without adhesion the pickee stays on the ground after the picker lifts;
    // with adhesion it follows the picker up.
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["friction"]["enable"] = true;
    // Slightly enlarge the contact active band: v1 adhesion is only applied
    // to pairs in the IPC friction band, so the band size limits the reach.
    // v2 adhesion will have its own active set independent of d_hat.
    config["contact"]["d_hat"]              = 0.02;
    // Relax strict mode so a momentary line-search hiccup during lift
    // (caused by pairs leaving/entering the active band) doesn't abort.
    config["extras"]["strict_mode"]["enable"] = false;

    test::Scene::dump_config(config, out_path);
    Scene scene{config};

    // ---- timeline (frames) ----
    // Slow timing so the bond can quasi-statically transition through the
    // active band as the gap grows.
    constexpr int press_start  = 5;    // start pressing down
    constexpr int contact_at   = 60;   // top reaches the bottom
    constexpr int hold_until   = 100;  // stay in contact this long
    constexpr int lift_until   = 200;  // lift fully up by here (slow)
    constexpr int max_frames   = 260;  // total sim length

    // Ground at y=0. Cube size = 0.3, half-extent = 0.15.
    // Pickee center at y=0.155 (slight gap above ground for IPC barrier zone).
    constexpr Float top_initial_y = 0.7;    // starting Y of top cube center
    constexpr Float top_contact_y = 0.470;  // touching pickee top face
    constexpr Float top_final_y   = 0.9;    // lifted Y (modest lift)

    {
        AffineBodyConstitution  abd;
        SoftTransformConstraint stc;

        auto& tabular         = scene.contact_tabular();
        auto  default_element = tabular.default_element();
        tabular.default_model(0.5, 1.0_GPa);

        if(adhesion_on)
        {
            RCCAdhesive adhesive;
            // Moderate Cn so adhesion is strong enough to lift the pickee
            // against gravity but soft enough that line search converges.
            adhesive.default_model(tabular,
                                   /*Cn=*/1e6,
                                   /*Ct=*/1e6,
                                   /*W=*/1.0,
                                   /*eta=*/2.0,
                                   /*bonding_rate=*/1.0,
                                   /*p0=*/0.0,
                                   /*initial_beta=*/1.0,
                                   /*enabled=*/true);
        }

        Transform pre = Transform::Identity();
        pre.scale(0.3);
        SimplicialComplexIO io{pre};

        auto              object = scene.objects().create("cubes");
        SimplicialComplex cube   = io.read(fmt::format("{}cube.msh", tetmesh_dir));
        label_surface(cube);
        label_triangle_orient(cube);

        cube.instances().resize(2);
        abd.apply_to(cube, 100.0_MPa);
        default_element.apply_to(cube);
        // The top cube is animated via SoftTransformConstraint.
        // High translation stiffness so it tracks the aim_transform tightly;
        // rotation stiffness 0 so we don't fight the body's natural orientation.
        // High translation stiffness so the picker tracks aim_transform
        // rigidly even when fighting adhesion + the pickee's inertia.
        stc.apply_to(cube, Vector2{1e8, 0});

        auto trans_view    = view(cube.transforms());
        auto is_fixed      = cube.instances().find<IndexT>(builtin::is_fixed);
        auto is_fixed_view = view(*is_fixed);

        // pickee (bottom cube): free, rests on the ground
        Transform t0     = Transform::Identity();
        t0.translation() = Vector3::UnitY() * 0.155;
        trans_view[0]    = t0.matrix();
        is_fixed_view[0] = 0;

        // picker (top cube): animated (driven by aim_transform via SoftTransformConstraint)
        Transform t1     = Transform::Identity();
        t1.translation() = Vector3::UnitY() * top_initial_y;
        trans_view[1]    = t1.matrix();
        is_fixed_view[1] = 0;

        object->geometries().create(cube);

        // Ground half-plane at y=0.
        ImplicitGeometry ground_geo = ground(0.0);
        auto             ground_obj = scene.objects().create("ground");
        ground_obj->geometries().create(ground_geo);

        // animator: only constrain the top cube (instance index 1)
        auto& animator = scene.animator();
        animator.insert(
            *object,
            [=](Animation::UpdateInfo& info)
            {
                auto geo_slots = info.geo_slots();
                auto geo       = geo_slots[0]->geometry().as<SimplicialComplex>();

                auto is_constrained =
                    geo->instances().find<IndexT>(builtin::is_constrained);
                auto is_constrained_view = view(*is_constrained);
                is_constrained_view[0]   = 0;  // bottom cube free
                is_constrained_view[1]   = 1;  // top cube animated

                auto aim_attr = geo->instances().find<Matrix4x4>(builtin::aim_transform);
                auto aim_view = view(*aim_attr);

                auto    f = info.frame();
                Float   y;
                if(f < press_start)
                    y = top_initial_y;
                else if(f < contact_at)
                {
                    Float t = Float(f - press_start) / Float(contact_at - press_start);
                    y       = top_initial_y + (top_contact_y - top_initial_y) * t;
                }
                else if(f < hold_until)
                    y = top_contact_y;
                else if(f < lift_until)
                {
                    Float t = Float(f - hold_until) / Float(lift_until - hold_until);
                    y       = top_contact_y + (top_final_y - top_contact_y) * t;
                }
                else
                    y = top_final_y;

                Transform t = Transform::Identity();
                t.translate(Vector3{0, y, 0});
                aim_view[1] = t.matrix();

                // also keep aim of bottom cube identity to avoid uninitialized state
                Transform i = Transform::Identity();
                aim_view[0] = i.matrix();
            });
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", out_path, 0));

    while(world.frame() < max_frames)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", out_path, world.frame()));
    }
}
