#include "mixed_scene_builders.h"
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/builtin/constants.h>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <array>
#include <string>

namespace uipc::bench::mixed
{
namespace
{
void build_wrecking_ball_scene(core::Scene& scene)
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;
    namespace fs = std::filesystem;

    const fs::path scene_file =
        fs::path{UIPC_PROJECT_DIR} / "apps/examples/wrecking_ball/wrecking_ball.json";
    std::ifstream ifs(scene_file);
    UIPC_ASSERT(ifs.good(), "Failed to open scene file: {}", scene_file.string());

    Json wrecking_ball_scene;
    ifs >> wrecking_ball_scene;

    AffineBodyConstitution abd;
    scene.constitution_tabular().insert(abd);
    auto default_contact = scene.contact_tabular().default_element();
    scene.contact_tabular().default_model(0.01, 20.0_GPa);

    const Float scale = 1.0;
    Transform   T     = Transform::Identity();
    T.scale(scale);
    SimplicialComplexIO io{T};

    const std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto cube = io.read(fmt::format("{}cube.msh", tetmesh_dir));
    auto ball = io.read(fmt::format("{}ball.msh", tetmesh_dir));
    auto link = io.read(fmt::format("{}link.msh", tetmesh_dir));

    auto cube_obj = scene.objects().create("cubes");
    auto ball_obj = scene.objects().create("balls");
    auto link_obj = scene.objects().create("links");

    abd.apply_to(cube, 10.0_MPa);
    label_surface(cube);
    label_triangle_orient(cube);

    abd.apply_to(ball, 10.0_MPa);
    label_surface(ball);
    label_triangle_orient(ball);

    abd.apply_to(link, 10.0_MPa);
    label_surface(link);
    label_triangle_orient(link);

    default_contact.apply_to(cube);
    default_contact.apply_to(ball);
    default_contact.apply_to(link);

    auto build_mesh = [&](const Json& j, Object& obj, const SimplicialComplex& mesh)
    {
        Vector3 position{0.0, 0.0, 0.0};
        if(auto pos_it = j.find("position"); pos_it != j.end())
        {
            position[0] = (*pos_it)[0].get<Float>();
            position[1] = (*pos_it)[1].get<Float>();
            position[2] = (*pos_it)[2].get<Float>();
        }

        Eigen::Quaternion<Float> Q = Eigen::Quaternion<Float>::Identity();
        if(auto rot_it = j.find("rotation"); rot_it != j.end())
        {
            Vector3 rotation;
            rotation[0] = (*rot_it)[0].get<Float>();
            rotation[1] = (*rot_it)[1].get<Float>();
            rotation[2] = (*rot_it)[2].get<Float>();

            rotation *= std::numbers::pi / 180.0;
            Q = AngleAxis(rotation.z(), Vector3::UnitZ())
                * AngleAxis(rotation.y(), Vector3::UnitY())
                * AngleAxis(rotation.x(), Vector3::UnitX());
        }

        IndexT is_fixed = 0;
        if(auto fixed_it = j.find("is_dof_fixed"); fixed_it != j.end())
            is_fixed = fixed_it->get<bool>() ? 1 : 0;

        position *= scale;
        Transform t = Transform::Identity();
        t.translate(position).rotate(Q);

        SimplicialComplex this_mesh     = mesh;
        view(this_mesh.transforms())[0] = t.matrix();
        auto is_fixed_attr              = this_mesh.instances().find<IndexT>(builtin::is_fixed);
        view(*is_fixed_attr)[0]         = is_fixed;
        obj.geometries().create(this_mesh);
    };

    for(const Json& obj : wrecking_ball_scene)
    {
        if(obj["mesh"] == "link.msh")
            build_mesh(obj, *link_obj, link);
        else if(obj["mesh"] == "cube.msh")
            build_mesh(obj, *cube_obj, cube);
        else if(obj["mesh"] == "ball.msh")
            build_mesh(obj, *ball_obj, ball);
    }

    auto ground_obj = scene.objects().create("ground");
    ground_obj->geometries().create(geometry::ground(-1.0));
}

Vector3 make_grid_position(IndexT i,
                           IndexT cols,
                           const Vector3& base,
                           Float sx,
                           Float sy,
                           Float sz)
{
    const IndexT row = i / cols;
    const IndexT col = i % cols;
    return Vector3{base.x() + sx * static_cast<Float>(col),
                   base.y() + sy * static_cast<Float>(row),
                   base.z() + sz * static_cast<Float>(row)};
}

void build_fem_heavy_scene(core::Scene& scene, bool with_ground_contact)
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    const std::string tetmesh_dir{AssetDir::tetmesh_path()};
    SimplicialComplexIO io;

    auto wheel_axle    = io.read(fmt::format("{}wheel_axle.msh", tetmesh_dir));
    auto simple_axle   = io.read(fmt::format("{}simple_axle.msh", tetmesh_dir));
    auto cylinder_hole = io.read(fmt::format("{}cylinder_hole.msh", tetmesh_dir));

    StableNeoHookean snh;
    auto parm = ElasticModuli::youngs_poisson(120.0_kPa, 0.48);

    auto prepare_mesh = [&](SimplicialComplex& mesh)
    {
        label_surface(mesh);
        label_triangle_orient(mesh);
        snh.apply_to(mesh, parm, 1e3);
    };

    prepare_mesh(wheel_axle);
    prepare_mesh(simple_axle);
    prepare_mesh(cylinder_hole);

    if(with_ground_contact)
    {
        scene.contact_tabular().default_model(0.5, 1.0_GPa);
        auto default_contact = scene.contact_tabular().default_element();
        default_contact.apply_to(wheel_axle);
        default_contact.apply_to(simple_axle);
        default_contact.apply_to(cylinder_hole);
    }

    auto append_instances = [&](std::string_view name,
                                const SimplicialComplex& src,
                                IndexT count,
                                IndexT cols,
                                const Vector3& base,
                                Float sx,
                                Float sy,
                                Float sz)
    {
        auto obj = scene.objects().create(std::string{name});
        for(IndexT i = 0; i < count; ++i)
        {
            auto pos = make_grid_position(i, cols, base, sx, sy, sz);
            SimplicialComplex mesh = src;

            // Bake instance translation into vertex positions to keep transform identity.
            auto pos_view = view(mesh.positions());
            std::ranges::transform(pos_view,
                                   pos_view.begin(),
                                   [&](const Vector3& v) -> Vector3 { return v + pos; });
            view(mesh.transforms())[0].setIdentity();

            obj->geometries().create(mesh);
        }
    };

    // 6 + 6 + 12 = 24 instances, ~57k tets with selected assets.
    append_instances("fem_heavy_wheel_axle",
                     wheel_axle,
                     6,
                     3,
                     Vector3{-14.0, 6.0, -8.0},
                     4.8,
                     0.8,
                     3.4);
    append_instances("fem_heavy_simple_axle",
                     simple_axle,
                     6,
                     3,
                     Vector3{-14.0, 6.3, 2.0},
                     4.5,
                     0.8,
                     3.2);
    append_instances("fem_heavy_cylinder_hole",
                     cylinder_hole,
                     12,
                     4,
                     Vector3{-14.0, 6.8, 11.5},
                     3.6,
                     0.75,
                     2.9);

    if(with_ground_contact)
    {
        auto ground_obj = scene.objects().create("ground");
        ground_obj->geometries().create(geometry::ground(-1.2));
    }
}
}  // namespace

std::string_view scenario_name(MixedScenario scenario)
{
    switch(scenario)
    {
        case MixedScenario::AbdGravity:
            return "abd_gravity";
        case MixedScenario::FemGravity:
            return "fem_gravity";
        case MixedScenario::FemGroundContact:
            return "fem_ground_contact";
        case MixedScenario::WreckingBall:
            return "wrecking_ball";
        case MixedScenario::FemHeavyNoContact:
            return "fem_heavy_nocontact";
        case MixedScenario::FemHeavyGroundContact:
            return "fem_heavy_ground_contact";
        default:
            return "unknown";
    }
}

Json make_mixed_config(MixedScenario scenario, const MixedConfigOptions& options)
{
    using namespace uipc;
    using namespace uipc::core;

    Json config = Scene::default_config();
    config["extras"]["strict_mode"]["enable"] = true;
    config["line_search"]["max_iter"]         = 8;
    config["newton"]["max_iter"]              = 16;
    config["linear_system"]["tol_rate"]       = 1e-3;

    if(scenario == MixedScenario::WreckingBall)
    {
        config["gravity"]                       = Vector3{0.0, -9.8, 0.0};
        config["contact"]["friction"]["enable"] = true;
        config["contact"]["enable"]             = true;
        config["contact"]["d_hat"]              = 0.01;
        config["collision_detection"]["method"] = "stackless_bvh";
    }
    else
    {
        switch(scenario)
        {
            case MixedScenario::AbdGravity:
                config["gravity"]           = Vector3{0.0, -9.8, 0.0};
                config["contact"]["enable"] = false;
                break;
            case MixedScenario::FemGravity:
                config["gravity"]           = Vector3{0.0, -9.8, 0.0};
                config["contact"]["enable"] = false;
                break;
            case MixedScenario::FemGroundContact:
                config["gravity"]                       = Vector3{0.0, -9.8, 0.0};
                config["contact"]["enable"]             = true;
                config["contact"]["friction"]["enable"] = false;
                break;
            case MixedScenario::FemHeavyNoContact:
                config["gravity"]           = Vector3{0.0, -9.8, 0.0};
                config["contact"]["enable"] = false;
                break;
            case MixedScenario::FemHeavyGroundContact:
                config["gravity"]                       = Vector3{0.0, -9.8, 0.0};
                config["contact"]["enable"]             = true;
                config["contact"]["friction"]["enable"] = false;
                config["contact"]["d_hat"]              = 0.01;
                break;
            default:
                break;
        }
    }

    auto& debug = config["extras"]["debug"];
    debug["dump_linear_system"] = options.dump_linear_system ? 1 : 0;
    debug["dump_solution_x"]    = options.dump_solution_x ? 1 : 0;
    debug["dump_surface"]       = options.dump_surface ? 1 : 0;

    return config;
}

void populate_mixed_scene(MixedScenario scenario, core::Scene& scene)
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    if(scenario == MixedScenario::WreckingBall)
    {
        build_wrecking_ball_scene(scene);
        return;
    }

    switch(scenario)
    {
        case MixedScenario::AbdGravity:
        {
            AffineBodyConstitution abd;
            auto                   object = scene.objects().create("abd");

            vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
            vector<Vector3>  Vs = {Vector3{0, 0, 1},
                                   Vector3{0, -1, 0},
                                   Vector3{-std::sqrt(3.0) / 2.0, 0, -0.5},
                                   Vector3{std::sqrt(3.0) / 2.0, 0, -0.5}};

            auto mesh = tetmesh(Vs, Ts);
            label_surface(mesh);
            label_triangle_orient(mesh);
            abd.apply_to(mesh, 100.0_MPa);
            object->geometries().create(mesh);
            break;
        }
        case MixedScenario::FemGravity:
        {
            StableNeoHookean snh;
            auto             object = scene.objects().create("fem");

            vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
            vector<Vector3>  Vs = {Vector3{0, 1, 0},
                                   Vector3{0, 0, 1},
                                   Vector3{-std::sqrt(3.0) / 2.0, 0, -0.5},
                                   Vector3{std::sqrt(3.0) / 2.0, 0, -0.5}};

            auto mesh = tetmesh(Vs, Ts);
            label_surface(mesh);
            label_triangle_orient(mesh);

            auto parm = ElasticModuli::youngs_poisson(100.0_kPa, 0.49);
            snh.apply_to(mesh, parm, 1e3);
            object->geometries().create(mesh);
            break;
        }
        case MixedScenario::FemGroundContact:
        {
            StableNeoHookean snh;
            auto             object = scene.objects().create("fem_contact");

            scene.contact_tabular().default_model(0.5, 1.0_GPa);

            std::string tetmesh_dir{AssetDir::tetmesh_path()};
            SimplicialComplexIO io;
            auto mesh = io.read(fmt::format("{}cube.msh", tetmesh_dir));
            label_surface(mesh);
            label_triangle_orient(mesh);

            auto parm = ElasticModuli::youngs_poisson(20.0_kPa, 0.49);
            snh.apply_to(mesh, parm);

            object->geometries().create(mesh);
            object->geometries().create(ground(-1.2));
            break;
        }
        case MixedScenario::FemHeavyNoContact:
            build_fem_heavy_scene(scene, false);
            break;
        case MixedScenario::FemHeavyGroundContact:
            build_fem_heavy_scene(scene, true);
            break;
        default:
            break;
    }
}
}  // namespace uipc::bench::mixed
