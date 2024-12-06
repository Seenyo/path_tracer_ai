#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include "../include/optix/pipeline.hpp"
#include "../include/optix/scene.hpp"
#include "../include/utils/image_utils.hpp"

// Error check/report helper for CUDA
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                       \
            std::cerr << "CUDA call '" << #call << "' failed: "           \
                      << cudaGetErrorString(error) << std::endl;          \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

int main() {
    try {
        std::cout << "Initializing CUDA..." << std::endl;
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));

        std::cout << "Creating scene..." << std::endl;
        // Create scene with a cube
        OptixScene scene;

        // Total number of triangles
        const int num_triangles = 12;

        // Define materials
        std::vector<Material> materials(num_triangles);

        // Front face (red diffuse) - material index 0
        materials[0] = {
                make_float3(0.8f, 0.2f, 0.2f),  // base_color
                0.0f,                           // metallic
                make_float3(0.0f),              // emission
                0.5f,                           // roughness
                make_float3(0.04f),             // f0 (default for dielectrics)
                1.5f,                           // ior
                LAMBERTIAN,                     // type
                0.0f                            // pad
        };

        // Back face (green diffuse) - material index 1
        materials[1] = {
                make_float3(0.2f, 0.8f, 0.2f),  // base_color
                0.0f,                           // metallic
                make_float3(0.0f),              // emission
                0.5f,                           // roughness
                make_float3(0.04f),             // f0
                1.5f,                           // ior
                LAMBERTIAN,                     // type
                0.0f                            // pad
        };

        // Left face (blue metal) - material index 2
        materials[2] = {
                make_float3(0.2f, 0.2f, 0.8f),  // base_color
                1.0f,                           // metallic
                make_float3(0.0f),              // emission
                0.1f,                           // roughness
                make_float3(0.95f),             // f0
                1.5f,                           // ior
                METAL,                          // type
                0.0f                            // pad
        };

        // Right face (gold metal) - material index 3
        materials[3] = {
                make_float3(1.0f, 0.86f, 0.57f),// base_color (gold)
                1.0f,                           // metallic
                make_float3(0.0f),              // emission
                0.2f,                           // roughness
                make_float3(1.0f, 0.86f, 0.57f),// f0 (gold)
                1.5f,                           // ior
                METAL,                          // type
                0.0f                            // pad
        };

        // Top face (glass) - material index 4
        materials[4] = {
                make_float3(1.0f),              // base_color
                0.0f,                           // metallic
                make_float3(0.0f),              // emission
                0.0f,                           // roughness
                make_float3(0.04f),             // f0
                1.5f,                           // ior
                DIELECTRIC,                     // type
                0.0f                            // pad
        };

        // Bottom face (emissive) - material index 5
        materials[5] = {
                make_float3(1.0f),              // base_color
                0.0f,                           // metallic
                make_float3(10.0f),             // emission (bright white)
                0.0f,                           // roughness
                make_float3(0.0f),              // f0
                1.0f,                           // ior
                EMISSIVE,                       // type
                0.0f                            // pad
        };

        // Add triangles to the scene with appropriate material indices

        // Front face (material index 0)
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, -1.0f),  // bottom left
                make_float3(1.0f, -1.0f, -1.0f),   // bottom right
                make_float3(1.0f, 1.0f, -1.0f),    // top right
                0                                   // material index
        );
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, -1.0f),  // bottom left
                make_float3(1.0f, 1.0f, -1.0f),    // top right
                make_float3(-1.0f, 1.0f, -1.0f),   // top left
                0                                   // material index
        );

        // Back face (material index 1)
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, 1.0f),   // bottom left
                make_float3(1.0f, 1.0f, 1.0f),     // top right
                make_float3(1.0f, -1.0f, 1.0f),    // bottom right
                1                                   // material index
        );
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, 1.0f),   // bottom left
                make_float3(-1.0f, 1.0f, 1.0f),    // top left
                make_float3(1.0f, 1.0f, 1.0f),     // top right
                1                                   // material index
        );

        // Left face (material index 2)
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, -1.0f),  // front bottom
                make_float3(-1.0f, 1.0f, -1.0f),   // front top
                make_float3(-1.0f, 1.0f, 1.0f),    // back top
                2                                   // material index
        );
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, -1.0f),  // front bottom
                make_float3(-1.0f, 1.0f, 1.0f),    // back top
                make_float3(-1.0f, -1.0f, 1.0f),   // back bottom
                2                                   // material index
        );

        // Right face (material index 3)
        scene.addTriangle(
                make_float3(1.0f, -1.0f, -1.0f),   // front bottom
                make_float3(1.0f, 1.0f, 1.0f),     // back top
                make_float3(1.0f, 1.0f, -1.0f),    // front top
                3                                   // material index
        );
        scene.addTriangle(
                make_float3(1.0f, -1.0f, -1.0f),   // front bottom
                make_float3(1.0f, -1.0f, 1.0f),    // back bottom
                make_float3(1.0f, 1.0f, 1.0f),     // back top
                3                                   // material index
        );

        // Top face (material index 4)
        scene.addTriangle(
                make_float3(-1.0f, 1.0f, -1.0f),   // front left
                make_float3(1.0f, 1.0f, -1.0f),    // front right
                make_float3(1.0f, 1.0f, 1.0f),     // back right
                4                                   // material index
        );
        scene.addTriangle(
                make_float3(-1.0f, 1.0f, -1.0f),   // front left
                make_float3(1.0f, 1.0f, 1.0f),     // back right
                make_float3(-1.0f, 1.0f, 1.0f),    // back left
                4                                   // material index
        );

        // Bottom face (material index 5)
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, -1.0f),  // front left
                make_float3(1.0f, -1.0f, 1.0f),    // back right
                make_float3(1.0f, -1.0f, -1.0f),   // front right
                5                                   // material index
        );
        scene.addTriangle(
                make_float3(-1.0f, -1.0f, -1.0f),  // front left
                make_float3(-1.0f, -1.0f, 1.0f),   // back left
                make_float3(1.0f, -1.0f, 1.0f),    // back right
                5                                   // material index
        );

        // Set up lights
        std::vector<Light> lights(1);
        lights[0] = {
                make_float3(0.0f, 2.0f, 0.0f),  // position (above cube)
                20.0f,                          // intensity
                make_float3(1.0f),              // color (white)
                0.5f                            // radius
        };

        // Upload materials and lights to scene
        scene.setMaterials(materials);
        scene.setLights(lights);

        std::cout << "Creating and initializing OptiX pipeline..." << std::endl;
        // Create and initialize OptiX pipeline
        PathTracerPipeline pipeline;
        pipeline.initialize(materials);

        std::cout << "Building acceleration structure..." << std::endl;
        // Build acceleration structure
        scene.buildAcceleration(pipeline.getContext());

        std::cout << "Setting up render parameters..." << std::endl;
        // Create output buffer
        const uint32_t width = 800;
        const uint32_t height = 600;
        const size_t buffer_size = width * height * sizeof(float3);

        // Allocate device memory for output
        void* d_color_buffer_ptr;
        CUDA_CHECK(cudaMalloc(&d_color_buffer_ptr, buffer_size));
        float3* d_color_buffer = reinterpret_cast<float3*>(d_color_buffer_ptr);

        // Initialize output buffer to black
        CUDA_CHECK(cudaMemset(d_color_buffer_ptr, 0, buffer_size));

        // Random number generator for initial seed
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> dis;

        // Set up launch parameters
        LaunchParams params = {};
        params.frame.color_buffer = d_color_buffer;
        params.frame.width = width;
        params.frame.height = height;
        params.frame.samples_per_pixel = 1;  // For debugging
        params.frame.max_bounces = 10;       // Increased from 5 to 10

        // Debug: Print launch parameters
        std::cout << "Launch Parameters:" << std::endl;
        std::cout << "  Image Size: " << width << "x" << height << std::endl;
        std::cout << "  Samples Per Pixel: " << params.frame.samples_per_pixel << std::endl;
        std::cout << "  Max Bounces: " << params.frame.max_bounces << std::endl;

        // Camera setup
        params.camera.position = make_float3(4.0f, 3.0f, -4.0f);
        params.camera.direction = normalize(make_float3(-1.0f, -0.5f, 1.0f));
        params.camera.up = make_float3(0.0f, 1.0f, 0.0f);
        params.camera.fov = make_float2(
                45.0f * static_cast<float>(M_PI) / 180.0f,  // horizontal FOV
                35.0f * static_cast<float>(M_PI) / 180.0f   // vertical FOV
        );

        // Background color
        params.miss.bg_color = make_float3(0.0f, 0.0f, 0.2f);  // Dark blue

        // Geometry data
        params.geometry.normals = reinterpret_cast<float3*>(scene.getNormalsBuffer());
        params.geometry.lights = reinterpret_cast<Light*>(scene.getLightsBuffer());
        params.geometry.num_lights = static_cast<unsigned int>(lights.size());

        // Random state
        params.random.seed = dis(gen);

        // Set traversable handle
        params.traversable = scene.getTraversableHandle();

        std::cout << "Rendering..." << std::endl;
        // Progressive rendering loop
        const int num_frames = 100;  // Accumulate 100 frames
        for (int frame = 0; frame < num_frames; ++frame) {
            params.frame_number = frame;
            pipeline.render(params);
        }

        std::cout << "Reading back result..." << std::endl;
        // Allocate host memory for output
        std::vector<float3> color_buffer(width * height);
        CUDA_CHECK(cudaMemcpy(
                color_buffer.data(),
                d_color_buffer,
                buffer_size,
                cudaMemcpyDeviceToHost
        ));

        // Find maximum value for auto-exposure
        float max_value = 0.0f;
        for (const auto& pixel : color_buffer) {
            max_value = std::max(max_value, std::max(pixel.x, std::max(pixel.y, pixel.z)));
        }
        std::cout << "Maximum pixel value: " << max_value << std::endl;

        std::cout << "Saving output..." << std::endl;
        // Convert float3 buffer to RGB8 and apply tone mapping
        std::vector<unsigned char> rgb_buffer(width * height * 3);
        for (size_t i = 0; i < width * height; ++i) {
            const float3& pixel = color_buffer[i];

            // Apply tone mapping
            float3 mapped;
            mapped.x = powf(pixel.x / (1.0f + pixel.x), 1.0f / 2.2f);
            mapped.y = powf(pixel.y / (1.0f + pixel.y), 1.0f / 2.2f);
            mapped.z = powf(pixel.z / (1.0f + pixel.z), 1.0f / 2.2f);

            // Convert to RGB8
            rgb_buffer[i * 3 + 0] = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, mapped.x)) * 255.99f);
            rgb_buffer[i * 3 + 1] = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, mapped.y)) * 255.99f);
            rgb_buffer[i * 3 + 2] = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, mapped.z)) * 255.99f);

            // Print some debug info for a few pixels
            if (i < 5 || i > width * height - 5) {
                std::cout << "Pixel " << i << ": "
                          << "HDR(" << pixel.x << ", " << pixel.y << ", " << pixel.z << ") -> "
                          << "LDR(" << (int)rgb_buffer[i * 3 + 0] << ", "
                          << (int)rgb_buffer[i * 3 + 1] << ", "
                          << (int)rgb_buffer[i * 3 + 2] << ")\n";
            }
        }

        // Save both PPM and PNG versions
        save_ppm("output.ppm", rgb_buffer, width, height);
        save_png("output", rgb_buffer, width, height);

        // Cleanup
        CUDA_CHECK(cudaFree(d_color_buffer));

        std::cout << "Render completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
