#include <app/app.h>
#include <numbers>
#include <uipc/uipc.h>
#include <uipc/common/enumerate.h>
#include <uipc/constitution/affine_body_constitution.h>

TEST_CASE("85_cuda_mixed_abd_ramp_friction_smoke", "[cuda_mixed][abd][friction]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    Engine engine{"cuda_mixed", output_path};
    World  world{engine};

    auto config                             = test::Scene::default_config();
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = true;
    config["contact"]["constitution"]       = "ipc";
    config["line_search"]["report_energy"]  = true;
    test::Scene::dump_config(config, output_path);

    Scene scene{config};
    {
        AffineBodyConstitution abd;

        auto& contact_tabular = scene.contact_tabular();
        contact_tabular.default_model(0.5, 1.0_GPa);
        auto default_element = contact_tabular.default_element();

        constexpr SizeT N = 8;
        auto friction_rate_step = 1.0 / (N - 1);

        vector<ContactElement> contact_elements(N);
        for(auto&& [i, e] : enumerate(contact_elements))
        {
            e = contact_tabular.create(fmt::format("element{}", i));
            contact_tabular.insert(e, default_element, friction_rate_step * i, 1.0_GPa);
        }

        auto cubes = scene.objects().create("cube");
        {
            Transform pre_transform = Transform::Identity();
            pre_transform.scale(0.3);
            SimplicialComplexIO io{pre_transform};

            auto cube = io.read(fmt::format("{}{}", tetmesh_dir, "cube.msh"));

            label_surface(cube);
            label_triangle_orient(cube);
            abd.apply_to(cube, 100.0_MPa);

            Float step    = 0.5;
            Float start_x = -step * (N - 1) / 2.0;

            for(SizeT i = 0; i < N; ++i)
            {
                SimplicialComplex this_cube = cube;
                contact_elements[i].apply_to(this_cube);

                auto      trans_view = view(this_cube.transforms());
                Transform t          = Transform::Identity();
                t.translate(Vector3{start_x + step * i, 1, -0.7});
                t.rotate(AngleAxis(30.0 * std::numbers::pi / 180.0, Vector3::UnitX()));

                trans_view[0] = t.matrix();
                cubes->geometries().create(this_cube);
            }
        }

        auto object_ramp = scene.objects().create("ramp");
        {
            Float   theta = 30.0 * std::numbers::pi / 180.0;
            Vector3 N     = Vector3{0, std::cos(theta), std::sin(theta)};
            Vector3 P     = Vector3{0, 0, 0};
            object_ramp->geometries().create(halfplane(P, N));
        }
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    for(int i = 0; i < 40; ++i)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
    }
}
