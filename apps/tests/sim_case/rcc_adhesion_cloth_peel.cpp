#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/constitution/rcc_adhesive.h>

// Demo: two cloth patches start apart, are brought together (press to bond),
// then one end of the top patch is lifted to peel it off the bottom.
// Bottom cloth has 4 fixed corners; top cloth is fully free except for an
// animated "pull tab" that drives its 4 corners through a press-hold-peel
// trajectory.
//
// Outputs:
//   output/tests/sim_case/rcc_adhesion_cloth_peel.cpp/peel_with_adhesion/
//   output/tests/sim_case/rcc_adhesion_cloth_peel.cpp/peel_without_adhesion/
TEST_CASE("rcc_adhesion_cloth_peel", "[rcc_adhesion][demo]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    bool        adhesion_on = false;
    std::string label;

    SECTION("peel_with_adhesion")
    {
        adhesion_on = true;
        label       = "peel_with_adhesion";
    }
    SECTION("peel_without_adhesion")
    {
        adhesion_on = false;
        label       = "peel_without_adhesion";
    }

    std::string out_path = fmt::format(
        "{}{}/", AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE), label);

    Engine engine{"cuda", out_path};
    World  world{engine};

    auto  config = test::Scene::default_config();
    Float dt     = 0.01;
    config["dt"] = dt;
    config["gravity"]                       = Vector3{0, 0, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = true;
    // v1 adhesion is bounded to the IPC active band; widen it slightly so the
    // bond can stretch a little before snapping.
    config["contact"]["d_hat"]                = 0.02;
    config["linear_system"]["tol_rate"]       = 1e-3;
    config["extras"]["strict_mode"]["enable"] = false;

    test::Scene::dump_config(config, out_path);
    Scene scene{config};

    // ---- cloth grid params ----
    constexpr int   N            = 12;     // (N+1)^2 verts per patch
    constexpr Float cloth_size   = 0.5;
    constexpr Float spacing      = cloth_size / N;
    constexpr Float bottom_y     = 0.0;

    // Top cloth starting positions:
    //   y_apart   : both ends well above the bottom (out of bond range)
    //   y_bonded  : both ends sitting on top of the bottom cloth (in band)
    //   y_lifted  : one end lifted up to peel
    constexpr Float y_apart   = 0.15;
    constexpr Float y_bonded  = 0.012;
    constexpr Float y_lifted  = 0.40;

    // ---- animation timeline (frames at dt=0.01) ----
    constexpr int press_until = 50;     // [0,press_until) bring top down
    constexpr int hold_until  = 110;    // [press_until,hold_until) press flat, let bond form
    constexpr int peel_until  = 250;    // [hold_until,peel_until) lift one end
    constexpr int max_frames  = 300;

    auto make_patch = [&](Float base_y) -> SimplicialComplex
    {
        vector<Vector3>  Vs;
        vector<Vector3i> Fs;
        for(int i = 0; i <= N; ++i)
            for(int j = 0; j <= N; ++j)
                Vs.push_back(Vector3{i * spacing - cloth_size / 2,
                                     base_y,
                                     j * spacing - cloth_size / 2});
        for(int i = 0; i < N; ++i)
            for(int j = 0; j < N; ++j)
            {
                int v00 = i * (N + 1) + j;
                int v10 = (i + 1) * (N + 1) + j;
                int v01 = i * (N + 1) + (j + 1);
                int v11 = (i + 1) * (N + 1) + (j + 1);
                Fs.push_back(Vector3i{v00, v10, v11});
                Fs.push_back(Vector3i{v00, v11, v01});
            }
        auto m = trimesh(Vs, Fs);
        label_surface(m);
        mesh_partition(m, 16);
        return m;
    };

    // Index helpers for the (N+1)x(N+1) grid.
    auto vid = [&](int i, int j) { return i * (N + 1) + j; };
    // The four corners (in grid coords) of the top cloth:
    //   c00 = (0,0)            -- low-x, low-z
    //   c0N = (0,N)            -- low-x, high-z
    //   cN0 = (N,0)            -- high-x, low-z      <-- "pull tab" (lifted)
    //   cNN = (N,N)            -- high-x, high-z     <-- "pull tab" (lifted)
    // We lift the entire high-x edge (i==N) so the cloth peels along z.
    const int N_grid = N;
    auto is_pull_edge = [N_grid](int i, int /*j*/) { return i == N_grid; };
    auto is_anchor_edge = [N_grid](int i, int /*j*/) { return i == 0; };

    {
        NeoHookeanShell        nhs;
        SoftPositionConstraint spc;

        auto& tabular         = scene.contact_tabular();
        auto  default_contact = tabular.default_element();
        tabular.default_model(/*friction_rate=*/0.0, /*resistance=*/1.0_MPa);

        if(adhesion_on)
        {
            RCCAdhesive adhesive;
            adhesive.default_model(tabular,
                                   /*Cn=*/1e4,
                                   /*Ct=*/1e4,
                                   /*W=*/1.0,
                                   /*eta=*/2.0,
                                   /*bonding_rate=*/1.0,
                                   /*p0=*/0.0,
                                   /*initial_beta=*/1.0,
                                   /*enabled=*/true);
        }

        auto parm = ElasticModuli2D::youngs_poisson(1.0_MPa, 0.49);

        // ----- bottom cloth (4 fixed corners) -----
        {
            auto object = scene.objects().create("cloth_bottom");
            auto patch  = make_patch(bottom_y);
            nhs.apply_to(patch, parm);
            default_contact.apply_to(patch);

            auto is_fixed_view = view(*patch.vertices().find<IndexT>(builtin::is_fixed));
            is_fixed_view[vid(0, 0)]         = 1;
            is_fixed_view[vid(0, N)]         = 1;
            is_fixed_view[vid(N, 0)]         = 1;
            is_fixed_view[vid(N, N)]         = 1;

            object->geometries().create(patch);
        }

        // ----- top cloth (animated edges: anchor edge & pull edge) -----
        {
            auto object = scene.objects().create("cloth_top");
            auto patch  = make_patch(y_apart);
            nhs.apply_to(patch, parm);
            default_contact.apply_to(patch);
            // Stiff position constraint so the animated edges track aim tightly.
            spc.apply_to(patch, 1e6);

            object->geometries().create(patch);

            // Capture rest x/z of every animated vertex so the animator can
            // build absolute aim_position values inside the lambda.
            std::vector<Vector3> rest(patch.vertices().size());
            {
                auto pos = patch.positions().view();
                for(SizeT k = 0; k < pos.size(); ++k)
                    rest[k] = pos[k];
            }

            auto& animator = scene.animator();
            animator.insert(
                *object,
                [=, rest = std::move(rest)](Animation::UpdateInfo& info) mutable
                {
                    auto geo_slots = info.geo_slots();
                    auto geo = geo_slots[0]->geometry().as<SimplicialComplex>();
                    auto is_constrained_view =
                        view(*geo->vertices().find<IndexT>(builtin::is_constrained));
                    auto aim_view =
                        view(*geo->vertices().find<Vector3>(builtin::aim_position));

                    auto    f = info.frame();
                    // Press phase: top cloth uniformly moves from y_apart -> y_bonded.
                    // Hold phase: top stays at y_bonded.
                    // Peel phase: anchor edge stays at y_bonded, pull edge lifts to y_lifted.
                    Float y_uniform_press;
                    if(f < press_until)
                    {
                        Float t = Float(f) / Float(press_until);
                        y_uniform_press = y_apart + (y_bonded - y_apart) * t;
                    }
                    else
                        y_uniform_press = y_bonded;

                    Float y_pull_edge;
                    if(f < hold_until)
                        y_pull_edge = y_uniform_press;
                    else if(f < peel_until)
                    {
                        Float t = Float(f - hold_until)
                                  / Float(peel_until - hold_until);
                        y_pull_edge = y_bonded + (y_lifted - y_bonded) * t;
                    }
                    else
                        y_pull_edge = y_lifted;

                    // Mark only the two animated edges (i==0 and i==N) as constrained.
                    for(int i = 0; i <= N_grid; ++i)
                        for(int j = 0; j <= N_grid; ++j)
                        {
                            int     k     = i * (N_grid + 1) + j;
                            bool    pull  = is_pull_edge(i, j);
                            bool    anch  = is_anchor_edge(i, j);
                            is_constrained_view[k] = (pull || anch) ? 1 : 0;
                            if(anch)
                                aim_view[k] = Vector3{rest[k].x(),
                                                      y_uniform_press,
                                                      rest[k].z()};
                            else if(pull)
                                aim_view[k] = Vector3{rest[k].x(),
                                                      y_pull_edge,
                                                      rest[k].z()};
                        }
                });
        }
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
