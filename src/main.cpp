#include <iostream>
#include <string>
#include <chrono>
#include <cxxopts.hpp>
#include "scene.hpp"
#include "camera.hpp"
#include "renderer.hpp"
#include "gpu/optix_renderer.hpp"

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        cxxopts::Options options("Path Tracer", "GPU-accelerated path tracer using OptiX");
        
        options.add_options()
            ("m,mode", "Rendering mode (cpu/gpu)", cxxopts::value<std::string>()->default_value("gpu"))
            ("w,width", "Image width", cxxopts::value<int>()->default_value("800"))
            ("h,height", "Image height", cxxopts::value<int>()->default_value("450"))
            ("s,samples", "Samples per pixel", cxxopts::value<int>()->default_value("100"))
            ("b,bounces", "Maximum ray bounces", cxxopts::value<int>()->default_value("5"))
            ("g,gamma", "Gamma correction value", cxxopts::value<float>()->default_value("2.2"))
            ("i,input", "Input OBJ file path", cxxopts::value<std::string>()->default_value("IronMan/IronMan.obj"))
            ("o,output", "Output image file path", cxxopts::value<std::string>()->default_value("output.png"))
            ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Get rendering settings
        std::string mode = result["mode"].as<std::string>();
        std::string inputFile = result["input"].as<std::string>();
        std::string outputFile = result["output"].as<std::string>();

        // Create scene and load model
        Scene scene;
        if (!scene.loadFromObj(inputFile)) {
            std::cerr << "Failed to load model: " << inputFile << std::endl;
            return -1;
        }

        // Set up camera
        Camera camera(
            glm::vec3(0.0f, 2.0f, 5.0f),    // position
            glm::vec3(0.0f, 1.8f, 0.0f),     // target
            glm::vec3(0.0f, 1.0f, 0.0f),     // up vector
            45.0f                            // FOV
        );

        // Create renderer settings
        if (mode == "cpu") {
            Renderer::Settings settings;
            settings.width = result["width"].as<int>();
            settings.height = result["height"].as<int>();
            settings.samplesPerPixel = result["samples"].as<int>();
            settings.maxBounces = result["bounces"].as<int>();
            settings.gamma = result["gamma"].as<float>();

            // Create and run CPU renderer
            Renderer renderer(settings);
            
            auto startTime = std::chrono::high_resolution_clock::now();
            renderer.render(scene, camera);
            auto endTime = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            std::cout << "\nRendering completed in " << duration.count() / 1000.0f << " seconds" << std::endl;
            
            renderer.saveImage(outputFile);
            
        } else if (mode == "gpu") {
            OptixRenderer::Settings settings;
            settings.width = result["width"].as<int>();
            settings.height = result["height"].as<int>();
            settings.samplesPerPixel = result["samples"].as<int>();
            settings.maxBounces = result["bounces"].as<int>();
            settings.gamma = result["gamma"].as<float>();

            // Create and run GPU renderer
            try {
                OptixRenderer renderer(settings);
                renderer.initialize();
                
                auto startTime = std::chrono::high_resolution_clock::now();
                
                renderer.uploadScene(scene);
                renderer.render(camera);
                
                auto endTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                std::cout << "\nRendering completed in " << duration.count() / 1000.0f << " seconds" << std::endl;
                
                renderer.saveImage(outputFile);
                
            } catch (const std::exception& e) {
                std::cerr << "GPU rendering failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU rendering..." << std::endl;
                
                // Fall back to CPU rendering
                Renderer::Settings cpuSettings;
                cpuSettings.width = settings.width;
                cpuSettings.height = settings.height;
                cpuSettings.samplesPerPixel = settings.samplesPerPixel;
                cpuSettings.maxBounces = settings.maxBounces;
                cpuSettings.gamma = settings.gamma;
                
                Renderer cpuRenderer(cpuSettings);
                cpuRenderer.render(scene, camera);
                cpuRenderer.saveImage(outputFile);
            }
        } else {
            std::cerr << "Invalid rendering mode. Use 'cpu' or 'gpu'." << std::endl;
            return -1;
        }

        std::cout << "Image saved as: " << outputFile << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
