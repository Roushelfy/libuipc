#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

TEST_CASE("80_cuda_mixed_abd_bdf2_smoke", "[cuda_mixed][abd][bdf]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    Engine engine{"cuda_mixed", output_path};
    World  world{engine};

    auto config                             = test::Scene::default_config();
    config["dt"]                            = 0.01;
    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["integrator"]["type"]            = "bdf2";
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = false;
    config["contact"]["constitution"]       = "ipc";
    test::Scene::dump_config(config, output_path);

    Scene scene{config};
    {
        AffineBodyConstitution abd;
        scene.contact_tabular().default_model(0.5, 1.0_GPa);

        auto obj = scene.objects().create("falling_abd");

        vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
        vector<Vector3>  Vs = {Vector3{0, 1, 0},
                               Vector3{0, 0, 1},
                               Vector3{-std::sqrt(3) / 2, 0, -0.5},
                               Vector3{std::sqrt(3) / 2, 0, -0.5}};

        std::transform(Vs.begin(),
                       Vs.end(),
                       Vs.begin(),
                       [](const Vector3& v) { return v * 0.3; });

        auto tet = tetmesh(Vs, Ts);
        label_surface(tet);
        label_triangle_orient(tet);
        abd.apply_to(tet, 100.0_MPa);

        auto trans = view(tet.transforms());
        trans[0](1, 3) += 0.6;

        obj->geometries().create(tet);
        obj->geometries().create(ground(0.0));
    }

    world.init(scene);
    REQUIRE(world.is_valid());

    auto abd_accessor = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(abd_accessor != nullptr);
    REQUIRE(abd_accessor->body_count() == 1);

    SimplicialComplex abd_state = abd_accessor->create_geometry();
    abd_state.instances().create<Matrix4x4>(builtin::transform);

    auto sample_y = [&]() -> Float {
        abd_accessor->copy_to(abd_state);
        return abd_state.transforms().view()[0](1, 3);
    };

    world.retrieve();
    Float initial_y = sample_y();
    Float min_y     = initial_y;

    while(world.frame() < 120)
    {
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
        min_y = std::min(min_y, sample_y());
    }

    REQUIRE(min_y < initial_y - 1e-3);
}
