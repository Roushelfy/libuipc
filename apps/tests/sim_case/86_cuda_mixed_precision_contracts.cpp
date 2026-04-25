#include <app/app.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <ranges>
#include <uipc/uipc.h>
#include <uipc/common/muda_memory_tracker.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_fixed_joint.h>
#include <uipc/constitution/affine_body_prismatic_joint.h>
#include <uipc/constitution/affine_body_revolute_joint.h>
#include <uipc/constitution/affine_body_shell.h>
#include <uipc/constitution/empty.h>
#include <uipc/constitution/external_articulation_constraint.h>
#include <uipc/constitution/hookean_spring.h>
#include <uipc/constitution/kirchhoff_rod_bending.h>
#include <uipc/constitution/particle.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/constitution/soft_vertex_triangle_stitch.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/geometry/utils/affine_body/transform.h>

namespace
{
using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
namespace fs = std::filesystem;

constexpr int kWarmupFrames  = 5;
constexpr int kMeasureFrames = 10;

fs::path contract_workspace(std::string_view name)
{
    auto root = fs::path(AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE));
    return root / std::string(name);
}

void require_cuda_sync()
{
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

void require_zero_steady_state_allocations(World& world)
{
    for(int i = 0; i < kWarmupFrames; ++i)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }

    require_cuda_sync();
    common::muda_memory_tracker_reset();

    for(int i = 0; i < kMeasureFrames; ++i)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }

    require_cuda_sync();
    auto snapshot = common::muda_memory_tracker_snapshot();
    INFO("alloc_calls=" << snapshot.alloc_calls << ", free_calls=" << snapshot.free_calls
                        << ", resize_calls=" << snapshot.resize_calls
                        << ", reserve_calls=" << snapshot.reserve_calls
                        << ", copy_calls=" << snapshot.copy_calls);
    CHECK(snapshot.alloc_calls == 0);
    CHECK(snapshot.free_calls == 0);
    CHECK(snapshot.resize_calls == 0);
    CHECK(snapshot.reserve_calls == 0);
}

template <typename BuildScene>
void run_zero_alloc_case(std::string_view name, Json config, BuildScene&& build_scene)
{
    auto output_path = contract_workspace(name);
    fs::create_directories(output_path);
    test::Scene::dump_config(config, output_path.string());

    Engine engine{"cuda_mixed", output_path.string()};
    World  world{engine};
    Scene  scene{config};

    build_scene(scene);

    world.init(scene);
    REQUIRE(world.is_valid());
    require_zero_steady_state_allocations(world);
}

void build_single_abd_tet(Scene& scene)
{
    AffineBodyConstitution abd;
    auto                   obj = scene.objects().create("falling_abd");

    vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
    vector<Vector3>  Vs = {Vector3{0, 1, 0},
                          Vector3{0, 0, 1},
                          Vector3{-std::sqrt(3.0) / 2.0, 0, -0.5},
                          Vector3{std::sqrt(3.0) / 2.0, 0, -0.5}};
    std::transform(Vs.begin(), Vs.end(), Vs.begin(), [](const Vector3& v)
                   { return v * 0.3; });

    auto tet = tetmesh(Vs, Ts);
    label_surface(tet);
    label_triangle_orient(tet);
    abd.apply_to(tet, 100.0_MPa);

    auto trans = view(tet.transforms());
    trans[0](1, 3) += 0.6;
    obj->geometries().create(tet);
}

Json linear_solver_selection_config()
{
    auto config                 = test::Scene::default_config();
    config["dt"]                = 0.01;
    config["gravity"]           = Vector3{0, -9.8, 0};
    config["integrator"]["type"] = "bdf2";
    config["contact"]["enable"] = false;
    return config;
}

void run_linear_solver_selection_case(std::string_view name, Json config)
{
    auto output_path = contract_workspace(name);
    fs::create_directories(output_path);
    test::Scene::dump_config(config, output_path.string());

    Engine engine{"cuda_mixed", output_path.string()};
    World  world{engine};
    Scene  scene{config};

    build_single_abd_tet(scene);

    world.init(scene);
    REQUIRE(world.is_valid());

    for(int i = 0; i < 2; ++i)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }
    require_cuda_sync();
}

void write_socu_ordering_report(const fs::path& path,
                                SizeT           block_size,
                                double          near_band_ratio = 1.0,
                                double          off_band_ratio  = 0.0)
{
    Json report = {
        {"selected",
         {{"ok", true},
          {"ordering",
           {{"orderer", "test"},
            {"block_size", block_size},
            {"chain_to_old", Json::array({0, 1, 2, 3})},
            {"old_to_chain", Json::array({0, 1, 2, 3})},
            {"atom_to_block", Json::array({0, 0, 0, 0})},
            {"atom_block_offset", Json::array({0, 3, 6, 9})},
            {"atom_dof_count", Json::array({3, 3, 3, 3})},
            {"block_to_atom_range",
             Json::array({Json{{"block", 0},
                                {"chain_begin", 0},
                                {"chain_end", 4},
                                {"dof_count", 12}}})}}},
          {"metrics",
           {{"near_band_ratio", near_band_ratio},
            {"off_band_ratio", off_band_ratio}}}}}};

    std::ofstream ofs{path};
    ofs << report.dump(2);
}

void require_socu_approx_init_failure(std::string_view name,
                                      Json             config,
                                      std::string_view expected_reason)
{
    auto output_path = contract_workspace(name);
    fs::create_directories(output_path);
    test::Scene::dump_config(config, output_path.string());

    Engine engine{"cuda_mixed", output_path.string()};
    World  world{engine};
    Scene  scene{config};
    build_single_abd_tet(scene);

    bool        threw = false;
    std::string message;
    try
    {
        world.init(scene);
    }
    catch(const std::exception& e)
    {
        threw   = true;
        message = e.what();
    }

    INFO("exception message: " << message);
    REQUIRE(threw);
    REQUIRE(message.find(expected_reason) != std::string::npos);
}
}  // namespace

TEST_CASE("86_cuda_mixed_contract_abd_bdf2_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][abd][bdf]")
{
    auto config                 = test::Scene::default_config();
    config["dt"]                = 0.01;
    config["gravity"]           = Vector3{0, -9.8, 0};
    config["integrator"]["type"] = "bdf2";
    config["contact"]["enable"] = false;

    run_zero_alloc_case("abd_bdf2_zero_alloc", config, build_single_abd_tet);
}

TEST_CASE("86_cuda_mixed_linear_solver_selection_smoke",
          "[cuda_mixed][contract][linear_solver]")
{
    SECTION("default_fused_pcg")
    {
        auto config = linear_solver_selection_config();
        run_linear_solver_selection_case("linear_solver_default_fused_pcg", config);
    }

    SECTION("explicit_linear_pcg")
    {
        auto config                       = linear_solver_selection_config();
        config["linear_system"]["solver"] = "linear_pcg";
        run_linear_solver_selection_case("linear_solver_explicit_linear_pcg", config);
    }

    SECTION("invalid_solver")
    {
        auto config                       = linear_solver_selection_config();
        config["linear_system"]["solver"] = "invalid_solver";
        auto output_path = contract_workspace("linear_solver_invalid_solver");
        fs::create_directories(output_path);
        test::Scene::dump_config(config, output_path.string());

        Engine engine{"cuda_mixed", output_path.string()};
        World  world{engine};
        Scene  scene{config};
        build_single_abd_tet(scene);

        REQUIRE_THROWS(world.init(scene));
    }

    SECTION("socu_approx_missing_ordering_report")
    {
        auto config                       = linear_solver_selection_config();
        config["linear_system"]["solver"] = "socu_approx";
        require_socu_approx_init_failure(
            "linear_solver_socu_approx_missing_ordering",
            config,
            "ordering_missing");
    }

    SECTION("socu_approx_rejects_bad_block_size")
    {
        auto config                       = linear_solver_selection_config();
        config["linear_system"]["solver"] = "socu_approx";
        auto output_path = contract_workspace("linear_solver_socu_approx_bad_block");
        fs::create_directories(output_path);
        auto report_path = output_path / "ordering.json";
        write_socu_ordering_report(report_path, 48);
        config["linear_system"]["socu_approx"]["ordering_report"] =
            report_path.string();
        require_socu_approx_init_failure(
            "linear_solver_socu_approx_bad_block",
            config,
            "unsupported_block_size");
    }

    SECTION("socu_approx_rejects_low_ordering_quality")
    {
        auto config                       = linear_solver_selection_config();
        config["linear_system"]["solver"] = "socu_approx";
        config["linear_system"]["socu_approx"]["min_near_band_ratio"] = 0.9;
        auto output_path = contract_workspace("linear_solver_socu_approx_low_quality");
        fs::create_directories(output_path);
        auto report_path = output_path / "ordering.json";
        write_socu_ordering_report(report_path, 32, 0.5, 0.5);
        config["linear_system"]["socu_approx"]["ordering_report"] =
            report_path.string();
        require_socu_approx_init_failure(
            "linear_solver_socu_approx_low_quality",
            config,
            "ordering_quality_too_low");
    }

    SECTION("socu_approx_valid_report_stops_at_m4_stub")
    {
        auto config                       = linear_solver_selection_config();
        config["linear_system"]["solver"] = "socu_approx";
        auto output_path = contract_workspace("linear_solver_socu_approx_stub");
        fs::create_directories(output_path);
        auto report_path = output_path / "ordering.json";
        write_socu_ordering_report(report_path, 32);
        config["linear_system"]["socu_approx"]["ordering_report"] =
            report_path.string();
        require_socu_approx_init_failure(
            "linear_solver_socu_approx_stub",
            config,
            "structured_provider_missing");
    }
}

TEST_CASE("86_cuda_mixed_contract_abd_fixed_joint_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][abd][joint]")
{
    auto config                            = test::Scene::default_config();
    config["gravity"]                      = Vector3{0, -9.8, 0};
    config["contact"]["enable"]            = false;
    config["line_search"]["report_energy"] = true;
    config["dt"]                           = 0.01;

    run_zero_alloc_case(
        "abd_fixed_joint_zero_alloc",
        config,
        [](Scene& scene)
        {
            Transform pre_transform = Transform::Identity();
            pre_transform.scale(0.3);
            SimplicialComplexIO io{pre_transform};

            AffineBodyConstitution  abd;
            SoftTransformConstraint stc;

            auto              left_object = scene.objects().create("left");
            SimplicialComplex left_mesh =
                io.read(fmt::format("{}cube.obj", AssetDir::trimesh_path()));
            abd.apply_to(left_mesh, 100.0_MPa);
            stc.apply_to(left_mesh, Vector2{100, 100});
            label_surface(left_mesh);
            {
                Transform t = Transform::Identity();
                t.translate(Vector3{-0.5, 0.5, 0});
                view(left_mesh.transforms())[0] = t.matrix();
            }
            auto [left_geo_slot, _l] = left_object->geometries().create(left_mesh);

            auto              right_object = scene.objects().create("right");
            SimplicialComplex right_mesh =
                io.read(fmt::format("{}cube.obj", AssetDir::trimesh_path()));
            abd.apply_to(right_mesh, 100.0_MPa);
            label_surface(right_mesh);
            {
                Transform t = Transform::Identity();
                t.translate(Vector3{0.5, 0.5, 0});
                view(right_mesh.transforms())[0] = t.matrix();
            }
            auto [right_geo_slot, _r] = right_object->geometries().create(right_mesh);

            SimplicialComplex                joint_mesh;
            AffineBodyFixedJoint             fixed_joint;
            vector<S<SimplicialComplexSlot>> l_geo_slots = {left_geo_slot};
            vector<S<SimplicialComplexSlot>> r_geo_slots = {right_geo_slot};
            fixed_joint.apply_to(joint_mesh, span{l_geo_slots}, span{r_geo_slots}, 100.0);

            auto joint_object = scene.objects().create("fixed_joint");
            joint_object->geometries().create(joint_mesh);

            scene.animator().insert(
                *left_object,
                [](Animation::UpdateInfo& info)
                {
                    auto geo_slots = info.geo_slots();
                    auto geo       = geo_slots[0]->geometry().as<SimplicialComplex>();

                    auto is_constrained = geo->instances().find<IndexT>(builtin::is_constrained);
                    view(*is_constrained)[0] = 1;

                    auto aim      = geo->instances().find<Matrix4x4>(builtin::aim_transform);
                    auto aim_view = view(*aim);

                    constexpr Float pi = std::numbers::pi;
                    Float           t  = info.frame() * info.dt();

                    Float x_offset = 0.5 * std::sin(2 * pi * t);
                    Float y_offset = 0.3 * std::sin(4 * pi * t);
                    Float theta    = pi / 4 * std::sin(2 * pi * t);

                    Transform transform = Transform::Identity();
                    transform.translate(Vector3{-0.5 + x_offset, 0.5 + y_offset, 0});
                    transform.rotate(Eigen::AngleAxisd(theta, Vector3::UnitZ()));
                    aim_view[0] = transform.matrix();
                });
        });
}

TEST_CASE("86_cuda_mixed_contract_external_articulation_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][abd][articulation]")
{
    auto config                 = test::Scene::default_config();
    config["gravity"]           = Vector3{0.0, -9.8, 0.0};
    config["contact"]["enable"] = false;

    run_zero_alloc_case(
        "external_articulation_zero_alloc",
        config,
        [](Scene& scene)
        {
            Transform pre_transform = Transform::Identity();
            pre_transform.scale(0.4);
            SimplicialComplexIO io{pre_transform};

            auto left_link  = scene.objects().create("left");
            auto right_link = scene.objects().create("right");

            AffineBodyConstitution abd;
            SimplicialComplex      abd_mesh =
                io.read(fmt::format("{}cube.obj", AssetDir::trimesh_path()));
            abd_mesh.instances().resize(2);
            label_surface(abd_mesh);
            abd.apply_to(abd_mesh, 100.0_MPa);

            SimplicialComplex left_mesh = abd_mesh;
            {
                Transform t0 = Transform::Identity();
                t0.translate(Vector3::UnitX() * -0.6 + Vector3::UnitZ() * -0.5);
                view(left_mesh.transforms())[0] = t0.matrix();

                Transform t1 = Transform::Identity();
                t1.translate(Vector3::UnitX() * -0.6 + Vector3::UnitZ() * 0.5);
                view(left_mesh.transforms())[1] = t1.matrix();

                auto is_fixed = left_mesh.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 0;
                view(*is_fixed)[1] = 0;

                auto ref_dof_prev = left_mesh.instances().create<Vector12>("ref_dof_prev");
                auto ref_dof_prev_view = view(*ref_dof_prev);
                auto transform_view    = left_mesh.transforms().view();
                std::ranges::transform(transform_view,
                                       ref_dof_prev_view.begin(),
                                       affine_body::transform_to_q);

                auto external_kinetic =
                    left_mesh.instances().find<IndexT>(builtin::external_kinetic);
                view(*external_kinetic)[0] = 1;
                view(*external_kinetic)[1] = 1;
            }
            auto [left_geo_slot, _left_rest] = left_link->geometries().create(left_mesh);

            scene.animator().insert(
                *left_link,
                [&](Animation::UpdateInfo& info)
                {
                    auto geo = info.geo_slots()[0]->geometry().as<SimplicialComplex>();
                    auto ref_dof_prev = geo->instances().find<Vector12>("ref_dof_prev");
                    auto ref_dof_prev_view = view(*ref_dof_prev);
                    auto transform_view    = geo->transforms().view();
                    std::ranges::transform(transform_view,
                                           ref_dof_prev_view.begin(),
                                           affine_body::transform_to_q);
                });

            SimplicialComplex right_mesh = abd_mesh;
            {
                Transform t0 = Transform::Identity();
                t0.translate(Vector3::UnitX() * 0.6 + Vector3::UnitZ() * -0.5);
                view(right_mesh.transforms())[0] = t0.matrix();

                Transform t1 = Transform::Identity();
                t1.translate(Vector3::UnitX() * 0.6 + Vector3::UnitZ() * 0.5);
                view(right_mesh.transforms())[1] = t1.matrix();

                auto is_fixed = right_mesh.instances().find<IndexT>(builtin::is_fixed);
                view(*is_fixed)[0] = 1;
                view(*is_fixed)[1] = 1;

                auto external_kinetic =
                    right_mesh.instances().find<IndexT>(builtin::external_kinetic);
                view(*external_kinetic)[0] = 1;
                view(*external_kinetic)[1] = 1;
            }
            auto [right_geo_slot, _right_rest] =
                right_link->geometries().create(right_mesh);

            AffineBodyRevoluteJoint abrj;
            vector<Vector2i>        Es = {{0, 1}, {2, 3}};
            vector<Vector3>         Vs = {{0, 0, 0.0},
                                  {0, 0, 1.0},
                                  {0, 0, -1.0},
                                  {0, 0, 0.0}};

            auto joint_mesh = linemesh(Vs, Es);
            label_surface(joint_mesh);

            vector<S<SimplicialComplexSlot>> l_geo_slots = {left_geo_slot, left_geo_slot};
            vector<IndexT>                   l_instance_id = {0, 1};
            vector<S<SimplicialComplexSlot>> r_geo_slots = {right_geo_slot, right_geo_slot};
            vector<IndexT>                   r_instance_id = {0, 1};
            vector<Float>                    strength_ratios = {100.0, 100.0};

            abrj.apply_to(joint_mesh,
                          l_geo_slots,
                          l_instance_id,
                          r_geo_slots,
                          r_instance_id,
                          strength_ratios);

            auto joint_object = scene.objects().create("joint_object");
            auto [joint_mesh_slot, _joint_rest] = joint_object->geometries().create(joint_mesh);

            ExternalArticulationConstraint eac;
            {
                vector<S<const GeometrySlot>> joint_geos = {joint_mesh_slot};
                vector<IndexT>                indices    = {0};
                auto articulation = eac.create_geometry(joint_geos, indices);
                auto mass         = articulation["joint_joint"]->find<Float>("mass");
                REQUIRE(mass);
                view(*mass)[0] = 64;

                auto articulation_object = scene.objects().create("articulation_object_1");
                articulation_object->geometries().create(articulation);

                scene.animator().insert(*articulation_object,
                                        [](Animation::UpdateInfo& info)
                                        {
                                            Float dt = info.dt();
                                            auto& geo = info.geo_slots()[0]->geometry();
                                            auto delta_theta_tilde =
                                                geo["joint"]->find<Float>("delta_theta_tilde");
                                            view(*delta_theta_tilde)[0] =
                                                -std::numbers::pi / 6 * dt;
                                        });
            }

            {
                vector<S<const GeometrySlot>> joint_geos = {joint_mesh_slot};
                vector<IndexT>                indices    = {1};
                auto articulation = eac.create_geometry(joint_geos, indices);
                auto mass         = articulation["joint_joint"]->find<Float>("mass");
                REQUIRE(mass);
                view(*mass)[0] = 64;

                auto articulation_object = scene.objects().create("articulation_object_2");
                articulation_object->geometries().create(articulation);

                scene.animator().insert(*articulation_object,
                                        [](Animation::UpdateInfo& info)
                                        {
                                            Float dt = info.dt();
                                            auto& geo = info.geo_slots()[0]->geometry();
                                            auto delta_theta_tilde =
                                                geo["joint"]->find<Float>("delta_theta_tilde");
                                            view(*delta_theta_tilde)[0] =
                                                std::numbers::pi / 6 * dt;
                                        });
            }
        });
}

TEST_CASE("86_cuda_mixed_contract_fem_mas_hybrid_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][fem][mas]")
{
    auto config                            = test::Scene::default_config();
    config["gravity"]                      = Vector3{0, -9.8, 0};
    config["contact"]["enable"]            = false;
    config["line_search"]["max_iter"]      = 8;
    config["linear_system"]["tol_rate"]    = 1e-3;

    run_zero_alloc_case(
        "fem_mas_hybrid_zero_alloc",
        config,
        [](Scene& scene)
        {
            std::string tetmesh_dir{AssetDir::tetmesh_path()};
            StableNeoHookean snh;

            auto object = scene.objects().create("hybrid");

            Matrix4x4 pre_trans = Matrix4x4::Identity();
            pre_trans(0, 0) = 0.2;
            pre_trans(1, 1) = 0.2;
            pre_trans(2, 2) = 0.2;
            SimplicialComplexIO scaled_io(pre_trans);

            auto make_cube = [&](double y_offset, bool with_partition)
            {
                auto cube = scaled_io.read(fmt::format("{}/cube.msh", tetmesh_dir));
                label_surface(cube);
                label_triangle_orient(cube);
                if(with_partition)
                {
                    mesh_partition(cube, 16);
                }
                snh.apply_to(cube, ElasticModuli::youngs_poisson(1e5, 0.49));
                auto pos = view(cube.positions());
                for(auto& p : pos)
                {
                    p[1] += y_offset;
                }
                object->geometries().create(cube);
            };

            make_cube(1.0, true);
            make_cube(1.7, false);
        });
}

TEST_CASE("86_cuda_mixed_contract_shell_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][abd][shell]")
{
    auto config                 = test::Scene::default_config();
    config["gravity"]           = Vector3{0, -9.8, 0};
    config["contact"]["enable"] = false;

    run_zero_alloc_case(
        "shell_zero_alloc",
        config,
        [](Scene& scene)
        {
            AffineBodyShell abd_shell;

            auto object = scene.objects().create("shell");
            vector<Vector3> Vs = {
                Vector3{-0.5, 0, -0.5},
                Vector3{0.5, 0, -0.5},
                Vector3{0.5, 0, 0.5},
                Vector3{-0.5, 0, 0.5}};
            vector<Vector3i> Fs = {Vector3i{0, 1, 2}, Vector3i{0, 2, 3}};

            auto mesh = trimesh(Vs, Fs);
            label_surface(mesh);
            mesh.instances().resize(2);
            abd_shell.apply_to(mesh, 100.0_MPa, 1e3, 0.01);

            auto trans_view    = view(mesh.transforms());
            auto is_fixed      = mesh.instances().find<IndexT>(builtin::is_fixed);
            auto is_fixed_view = view(*is_fixed);

            Transform t0 = Transform::Identity();
            t0.translation() = Vector3{0, 1.0, 0};
            trans_view[0]    = t0.matrix();
            is_fixed_view[0] = 0;

            Transform t1 = Transform::Identity();
            t1.translation() = Vector3{1.5, 1.0, 0};
            trans_view[1]    = t1.matrix();
            is_fixed_view[1] = 1;

            object->geometries().create(mesh);
        });
}

TEST_CASE("86_cuda_mixed_contract_rod_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][fem][mas][rod]")
{
    auto config                             = test::Scene::default_config();
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = false;
    config["contact"]["friction"]["enable"] = false;
    config["linear_system"]["tol_rate"]     = 1e-3;

    run_zero_alloc_case(
        "rod_zero_alloc",
        config,
        [](Scene& scene)
        {
            HookeanSpring       hs;
            KirchhoffRodBending krb;

            auto object = scene.objects().create("rods");
            constexpr int n = 20;
            constexpr int num_rods = 3;

            for(int r = 0; r < num_rods; ++r)
            {
                vector<Vector3> Vs(n);
                for(int i = 0; i < n; ++i)
                {
                    Vs[i] = Vector3{r * 0.1, 0.3, i * 0.03};
                }

                vector<Vector2i> Es(n - 1);
                for(int i = 0; i < n - 1; ++i)
                {
                    Es[i] = Vector2i{i, i + 1};
                }

                auto mesh = linemesh(Vs, Es);
                label_surface(mesh);
                mesh_partition(mesh, 16);

                hs.apply_to(mesh, 10.0_MPa);
                krb.apply_to(mesh, 1.0_MPa);

                auto is_fixed      = mesh.vertices().find<IndexT>(builtin::is_fixed);
                auto is_fixed_view = view(*is_fixed);
                is_fixed_view[0] = 1;
                is_fixed_view[1] = 1;

                object->geometries().create(mesh);
            }
        });
}

TEST_CASE("86_cuda_mixed_contract_stitch_zero_alloc",
          "[cuda_mixed][contract][steady_alloc][inter_primitive][stitch]")
{
    auto config                 = test::Scene::default_config();
    config["gravity"]           = Vector3{0, -9.8, 0};
    config["contact"]["enable"] = false;

    run_zero_alloc_case(
        "stitch_zero_alloc",
        config,
        [](Scene& scene)
        {
            Empty    empty;
            Particle particle;

            Float Y = 1.0;
            vector<Vector3> vert_pos = {{0.25, Y - 0.5, 0.25},
                                        {0.5, Y - 0.3, 0.1},
                                        {0.1, Y - 0.4, 0.5},
                                        {0.3, Y, 0.3}};
            auto vertex_mesh = pointcloud(vert_pos);
            particle.apply_to(vertex_mesh);
            label_surface(vertex_mesh);

            vector<Vector3>  tri_pos = {{0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}};
            vector<Vector3i> tri_faces = {{0, 1, 2}};
            auto             triangle_mesh = trimesh(tri_pos, tri_faces);
            empty.apply_to(triangle_mesh);
            label_surface(triangle_mesh);
            auto is_fixed = triangle_mesh.vertices().find<IndexT>(builtin::is_fixed);
            std::ranges::fill(view(*is_fixed), 1);

            auto vert_obj = scene.objects().create("vertex_provider");
            auto [vert_geo_slot, vert_rest_slot] = vert_obj->geometries().create(vertex_mesh);

            auto tri_obj = scene.objects().create("triangle_provider");
            auto [tri_geo_slot, tri_rest_slot] = tri_obj->geometries().create(triangle_mesh);

            auto pairs_geo = closest_vertex_triangle_pairs(vertex_mesh, triangle_mesh, 1.0);
            REQUIRE(pairs_geo.instances().size() == 4);

            SoftVertexTriangleStitch stitch;
            auto stitch_geo = stitch.create_geometry({vert_geo_slot, tri_geo_slot},
                                                     {vert_rest_slot, tri_rest_slot},
                                                     pairs_geo,
                                                     ElasticModuli::youngs_poisson(120.0_kPa, 0.49),
                                                     0.1);

            auto stitch_obj = scene.objects().create("stitch");
            stitch_obj->geometries().create(stitch_geo);
        });
}

TEST_CASE("86_cuda_mixed_external_articulation_multijoints_smoke",
          "[cuda_mixed][abd][joint][articulation]")
{
    auto output_path = contract_workspace("external_articulation_multijoints_smoke");
    fs::create_directories(output_path);

    Engine engine{"cuda_mixed", output_path.string()};
    World  world{engine};

    auto config                 = test::Scene::default_config();
    config["gravity"]           = Vector3{0.0, -9.8, 0.0};
    config["contact"]["enable"] = false;
    test::Scene::dump_config(config, output_path.string());

    Scene scene{config};
    {
        Transform pre_transform = Transform::Identity();
        pre_transform.scale(0.4);
        SimplicialComplexIO io{pre_transform};

        auto links = scene.objects().create("links");

        AffineBodyConstitution abd;
        SimplicialComplex      abd_mesh =
            io.read(fmt::format("{}cube.obj", AssetDir::trimesh_path()));
        abd_mesh.instances().resize(3);
        label_surface(abd_mesh);
        abd.apply_to(abd_mesh, 100.0_MPa);

        Transform t0 = Transform::Identity();
        t0.translate(Vector3::UnitZ() * -0.8);
        view(abd_mesh.transforms())[0] = t0.matrix();

        Transform t1 = Transform::Identity();
        t1.translate(Vector3::UnitZ() * 0.0);
        view(abd_mesh.transforms())[1] = t1.matrix();

        Transform t2 = Transform::Identity();
        t2.translate(Vector3::UnitZ() * 0.8);
        view(abd_mesh.transforms())[2] = t2.matrix();

        auto is_fixed      = abd_mesh.instances().find<IndexT>(builtin::is_fixed);
        auto is_fixed_view = view(*is_fixed);
        is_fixed_view[0]   = 1;
        is_fixed_view[1]   = 0;
        is_fixed_view[2]   = 0;

        auto ref_dof_prev = abd_mesh.instances().create<Vector12>("ref_dof_prev");
        auto ref_dof_prev_view = view(*ref_dof_prev);
        auto transform_view    = abd_mesh.transforms().view();
        std::ranges::transform(
            transform_view, ref_dof_prev_view.begin(), affine_body::transform_to_q);

        auto external_kinetic = abd_mesh.instances().find<IndexT>(builtin::external_kinetic);
        auto external_kinetic_view = view(*external_kinetic);
        std::ranges::fill(external_kinetic_view, 1);

        auto [geo_slot, _rest_geo_slot] = links->geometries().create(abd_mesh);

        scene.animator().insert(
            *links,
            [&](Animation::UpdateInfo& info)
            {
                auto geo = info.geo_slots()[0]->geometry().as<SimplicialComplex>();
                auto ref_dof_prev = geo->instances().find<Vector12>("ref_dof_prev");
                auto ref_dof_prev_view = view(*ref_dof_prev);
                auto transform_view    = geo->transforms().view();
                std::ranges::transform(transform_view,
                                       ref_dof_prev_view.begin(),
                                       affine_body::transform_to_q);
            });

        S<SimplicialComplexSlot> revolute_slot;
        {
            AffineBodyRevoluteJoint abrj;
            vector<Vector2i>        Es = {{0, 1}};
            vector<Vector3>         Vs = {{-0.5, 0.0, -0.4}, {0.5, 0.0, -0.4}};
            auto                    joint_mesh = linemesh(Vs, Es);

            vector<S<SimplicialComplexSlot>> l_geo_slots   = {geo_slot};
            vector<IndexT>                   l_instance_id = {0};
            vector<S<SimplicialComplexSlot>> r_geo_slots   = {geo_slot};
            vector<IndexT>                   r_instance_id = {1};
            vector<Float>                    strength_ratios = {100.0};

            abrj.apply_to(
                joint_mesh, l_geo_slots, l_instance_id, r_geo_slots, r_instance_id, strength_ratios);

            auto joints = scene.objects().create("joints");
            auto [revolute_joint_slot, _rest] = joints->geometries().create(joint_mesh);
            revolute_slot = revolute_joint_slot;
        }

        S<SimplicialComplexSlot> prismatic_slot;
        {
            AffineBodyPrismaticJoint abpj;
            vector<Vector2i>         Es = {{0, 1}};
            vector<Vector3>          Vs = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.4}};
            auto                     joint_mesh = linemesh(Vs, Es);

            vector<S<SimplicialComplexSlot>> l_geo_slots   = {geo_slot};
            vector<IndexT>                   l_instance_id = {1};
            vector<S<SimplicialComplexSlot>> r_geo_slots   = {geo_slot};
            vector<IndexT>                   r_instance_id = {2};
            vector<Float>                    strength_ratios = {100.0};
            abpj.apply_to(
                joint_mesh, l_geo_slots, l_instance_id, r_geo_slots, r_instance_id, strength_ratios);

            auto joints = scene.objects().create("joints_prismatic");
            auto [prismatic_joint_slot, _rest] = joints->geometries().create(joint_mesh);
            prismatic_slot = prismatic_joint_slot;
        }

        ExternalArticulationConstraint eac;
        vector<S<const GeometrySlot>>  joint_geos = {revolute_slot, prismatic_slot};
        vector<IndexT>                 indices    = {0, 0};
        auto articulation = eac.create_geometry(joint_geos, indices);
        auto mass         = articulation["joint_joint"]->find<Float>("mass");
        REQUIRE(mass);
        auto mass_view = view(*mass);
        Eigen::Map<MatrixX> mass_mat = Eigen::Map<MatrixX>(mass_view.data(), 2, 2);
        mass_mat       = MatrixX::Identity(2, 2) * 64;
        mass_mat(0, 1) = 8;
        mass_mat(1, 0) = 8;

        auto articulation_object = scene.objects().create("articulation_object");
        articulation_object->geometries().create(articulation);

        scene.animator().insert(*articulation_object,
                                [](Animation::UpdateInfo& info)
                                {
                                    Float dt = info.dt();
                                    auto& geo = info.geo_slots()[0]->geometry();
                                    auto delta_theta_tilde =
                                        geo["joint"]->find<Float>("delta_theta_tilde");
                                    auto delta_theta_view = view(*delta_theta_tilde);
                                    delta_theta_view[0] = std::numbers::pi / 6 * dt;
                                    delta_theta_view[1] = 0.1 * dt;
                                });
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    for(int i = 0; i < 20; ++i)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }
}
