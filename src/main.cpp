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

        // Setup camera with better position to see lighting
        Camera camera(
            glm::vec3(3.0f, 2.0f, 3.0f),     // Position - moved back and up for better view
            glm::vec3(0.0f, 0.5f, 0.0f),     // Look at - slightly up from origin
            glm::vec3(0.0f, 1.0f, 0.0f),     // Up vector
            45.0f,                            // FOV - narrower for less distortion
            16.0f/9.0f,                       // Aspect ratio
            0.1f,                             // Aperture - slight DOF
            4.0f                              // Focus distance
        );

        // Setup renderer with higher quality settings
        Renderer::Settings settings;
        settings.width = 1920;                // Full HD resolution
        settings.height = 1080;               // 16:9 aspect ratio
        settings.samplesPerPixel = 100;       // More samples for better quality
        settings.maxBounces = 5;              // More bounces for better global illumination
        settings.gamma = 2.2f;                // Standard gamma correction

        // Create preview settings for quick test
        bool previewMode = true;  // Set to false for final render
        if (previewMode) {
            settings.width = 960;             // Half resolution
            settings.height = 540;
            settings.samplesPerPixel = 10;    // Fewer samples
            settings.maxBounces = 3;          // Fewer bounces
        }

        Renderer renderer(settings);

        // Start timing
        auto startTime = std::chrono::high_resolution_clock::now();

        // Render
        std::cout << "\nStarting render with settings:" << std::endl;
        std::cout << "Resolution: " << settings.width << "x" << settings.height << std::endl;
        std::cout << "Samples per pixel: " << settings.samplesPerPixel << std::endl;
        std::cout << "Max bounces: " << settings.maxBounces << std::endl;
        std::cout << "Preview mode: " << (previewMode ? "Yes" : "No") << std::endl;

        renderer.render(scene, camera);

        // End timing
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        // Save image
        std::string outputFile = previewMode ? "preview.png" : "final_render.png";
        std::cout << "\nSaving image as '" << outputFile << "'..." << std::endl;
        renderer.saveImage(outputFile);

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
