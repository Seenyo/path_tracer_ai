#include <iostream>
#include <chrono>
#include "../include/scene.hpp"
#include "../include/camera.hpp"
#include "../include/renderer.hpp"

int main() {
    try {
        // Create scene and load model
        Scene scene;
        if (!scene.loadFromObj("IronMan/IronMan.obj")) {
            std::cerr << "Failed to load model" << std::endl;
            return -1;
        }

        std::cout << "\nSetting up camera..." << std::endl;
        // Setup camera with optimal position for scaled and rotated Ironman model
        Camera camera(
            glm::vec3(0.0f, 3.f, -1.f),     // Position - diagonal view from front
            glm::vec3(0.0f, 2.75f, 0.0f),     // Look at - centered on model's chest
            glm::vec3(0.0f, -1.0f, 0.0f),     // Up vector
            45.0f,                            // FOV - narrower for less distortion
            21.0f/9.0f,                       // Aspect ratio
            1e-5f,                            // Aperture - small for depth of field
            3.5f                              // Focus distance - adjusted for room size
        );

        std::cout << "Setting up renderer..." << std::endl;
        // Setup renderer with high quality settings
        Renderer::Settings settings;
        settings.width = 3440;                // Full HD resolution
        settings.height = 1440;
        settings.samplesPerPixel = 1;       // More samples for better quality
        settings.maxBounces = 16;              // More bounces for better reflections
        settings.gamma = 2.2f;                // Standard gamma correction

        Renderer renderer(settings);

        // Start timing
        auto startTime = std::chrono::high_resolution_clock::now();

        std::cout << "\nStarting render..." << std::endl;
        renderer.render(scene, camera);

        // End timing
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        // Save image
        std::cout << "\nSaving final render..." << std::endl;
        renderer.saveImage("final_rendering.png");

        float renderTime = duration.count() / 1000.0f;
        std::cout << "\nRender completed in " << renderTime << " seconds" << std::endl;
        std::cout << "Average time per sample: " << 
            (renderTime / (settings.width * settings.height * settings.samplesPerPixel)) * 1000.0f << 
            " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
