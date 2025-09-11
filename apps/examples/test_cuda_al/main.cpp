#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <iostream>

int main()
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;
    namespace fs = std::filesystem;

    std::cout << "Testing CUDA AL Backend..." << std::endl;

    try {
        // Test cuda_al backend
        Engine engine{"cuda_al"};
        std::cout << "✅ Successfully created cuda_al Engine" << std::endl;

        World world{engine};
        std::cout << "✅ Successfully created World with cuda_al backend" << std::endl;

        auto config = Scene::default_config();
        config["gravity"] = Vector3{0, -9.8, 0};
        config["dt"] = 0.01_s;

        // Add contact configuration for IPC (same as original)
        config["contact"]["constitution"] = "ipc";

        Scene scene{config};
        std::cout << "✅ Successfully created Scene" << std::endl;

        {
            // create constitution and contact model
            AffineBodyConstitution abd;
            scene.constitution_tabular().insert(abd);

            // friction ratio and contact resistance
            scene.contact_tabular().default_model(0.5, 1.0_GPa);
            auto default_element = scene.contact_tabular().default_element();

            // create a simple tetrahedron (smaller simulation for testing)
            vector<Vector3> Vs = {Vector3{0, 1, 0},
                                  Vector3{0, 0, 1},
                                  Vector3{-std::sqrt(3) / 2, 0, -0.5},
                                  Vector3{std::sqrt(3) / 2, 0, -0.5}};
            vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};

            SimplicialComplex base_mesh = tetmesh(Vs, Ts);
            abd.apply_to(base_mesh, 100.0_MPa);
            default_element.apply_to(base_mesh);

            label_surface(base_mesh);
            label_triangle_orient(base_mesh);

            SimplicialComplex mesh1 = base_mesh;
            {
                // move the mesh1 up for 1 unit
                auto pos_view = view(mesh1.positions());
                std::ranges::transform(pos_view,
                                       pos_view.begin(),
                                       [](const Vector3& v) -> Vector3
                                       { return v + Vector3::UnitY() * 1.5; });
            }

            SimplicialComplex mesh2 = base_mesh;
            {
                // fix the bottom tetrahedron
                auto is_fixed = mesh2.instances().find<IndexT>(builtin::is_fixed);
                auto is_fixed_view = view(*is_fixed);
                is_fixed_view[0] = 1;
            }

            // create object with two geometries
            auto object = scene.objects().create("test_tets");
            {
                object->geometries().create(mesh1);
                object->geometries().create(mesh2);
            }
        }

        std::cout << "✅ Successfully set up scene geometry" << std::endl;

        world.init(scene);
        std::cout << "✅ Successfully initialized world with cuda_al backend" << std::endl;

        SceneIO sio{scene};
        auto this_output_path = AssetDir::output_path(__FILE__);

        // Run a few simulation steps to test functionality
        std::cout << "🚀 Running simulation with cuda_al backend..." << std::endl;
        
        // Write initial state
        sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));
        
        for(int i = 1; i <= 5; i++)
        {
            std::cout << "  Step " << i << "/5..." << std::endl;
            
            world.advance();
            world.sync();
            world.retrieve();
            
            // Write output for each step
            sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, i));
            
            std::cout << "    ✅ Step " << i << " completed successfully" << std::endl;
        }
        
        std::cout << "🎉 CUDA AL Backend test completed successfully!" << std::endl;
        std::cout << "   - Engine initialization: ✅" << std::endl;
        std::cout << "   - World creation: ✅" << std::endl;
        std::cout << "   - Scene setup: ✅" << std::endl;
        std::cout << "   - Simulation steps: ✅" << std::endl;
        std::cout << "   - Output files: " << this_output_path << "scene_surface*.obj" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}