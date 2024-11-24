#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include "../include/optix/pipeline.hpp"
#include "../include/optix/scene.hpp"
#include "../include/math/vec_math.hpp"
#include "../include/utils/image_utils.hpp"

// Error check/report helper for CUDA
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                      \
            std::cerr << "CUDA call '" << #call << "' failed: "          \
                     << cudaGetErrorString(error) << std::endl;          \
            exit(1);                                                     \
        }                                                                \
    } while (0)

int main() {
    try {
        std::cout << "Initializing CUDA..." << std::endl;
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));

        std::cout << "Creating scene..." << std::endl;
        // Create scene with a cube
        OptixScene scene;
        
        // Front face
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, -1.0f),  // bottom left
            make_float3(1.0f, -1.0f, -1.0f),   // bottom right
            make_float3(1.0f, 1.0f, -1.0f)     // top right
        );
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, -1.0f),  // bottom left
            make_float3(1.0f, 1.0f, -1.0f),    // top right
            make_float3(-1.0f, 1.0f, -1.0f)    // top left
        );

        // Back face
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, 1.0f),   // bottom left
            make_float3(1.0f, 1.0f, 1.0f),     // top right
            make_float3(1.0f, -1.0f, 1.0f)     // bottom right
        );
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, 1.0f),   // bottom left
            make_float3(-1.0f, 1.0f, 1.0f),    // top left
            make_float3(1.0f, 1.0f, 1.0f)      // top right
        );

        // Left face
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, -1.0f),  // front bottom
            make_float3(-1.0f, 1.0f, -1.0f),   // front top
            make_float3(-1.0f, 1.0f, 1.0f)     // back top
        );
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, -1.0f),  // front bottom
            make_float3(-1.0f, 1.0f, 1.0f),    // back top
            make_float3(-1.0f, -1.0f, 1.0f)    // back bottom
        );

        // Right face
        scene.addTriangle(
            make_float3(1.0f, -1.0f, -1.0f),   // front bottom
            make_float3(1.0f, 1.0f, 1.0f),     // back top
            make_float3(1.0f, 1.0f, -1.0f)     // front top
        );
        scene.addTriangle(
            make_float3(1.0f, -1.0f, -1.0f),   // front bottom
            make_float3(1.0f, -1.0f, 1.0f),    // back bottom
            make_float3(1.0f, 1.0f, 1.0f)      // back top
        );

        // Top face
        scene.addTriangle(
            make_float3(-1.0f, 1.0f, -1.0f),   // front left
            make_float3(1.0f, 1.0f, -1.0f),    // front right
            make_float3(1.0f, 1.0f, 1.0f)      // back right
        );
        scene.addTriangle(
            make_float3(-1.0f, 1.0f, -1.0f),   // front left
            make_float3(1.0f, 1.0f, 1.0f),     // back right
            make_float3(-1.0f, 1.0f, 1.0f)     // back left
        );

        // Bottom face
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, -1.0f),  // front left
            make_float3(1.0f, -1.0f, 1.0f),    // back right
            make_float3(1.0f, -1.0f, -1.0f)    // front right
        );
        scene.addTriangle(
            make_float3(-1.0f, -1.0f, -1.0f),  // front left
            make_float3(-1.0f, -1.0f, 1.0f),   // back left
            make_float3(1.0f, -1.0f, 1.0f)     // back right
        );

        // Set up materials
        std::vector<Material> materials(12);  // 12 triangles total

        // Front face (red diffuse)
        materials[0] = materials[1] = {
            make_float3(0.8f, 0.2f, 0.2f),  // base_color
            0.0f,                           // metallic
            make_float3(0.0f),              // emission
            0.5f,                           // roughness
            make_float3(0.04f),             // f0 (default for dielectrics)
            1.5f,                           // ior
            LAMBERTIAN                      // type
        };

        // Back face (green diffuse)
        materials[2] = materials[3] = {
            make_float3(0.2f, 0.8f, 0.2f),  // base_color
            0.0f,                           // metallic
            make_float3(0.0f),              // emission
            0.5f,                           // roughness
            make_float3(0.04f),             // f0
            1.5f,                           // ior
            LAMBERTIAN                      // type
        };

        // Left face (blue metal)
        materials[4] = materials[5] = {
            make_float3(0.2f, 0.2f, 0.8f),  // base_color
            1.0f,                           // metallic
            make_float3(0.0f),              // emission
            0.1f,                           // roughness
            make_float3(0.95f),             // f0
            1.5f,                           // ior
            METAL                           // type
        };

        // Right face (gold metal)
        materials[6] = materials[7] = {
            make_float3(1.0f, 0.8f, 0.2f),  // base_color
            1.0f,                           // metallic
            make_float3(0.0f),              // emission
            0.2f,                           // roughness
            make_float3(1.0f, 0.86f, 0.57f),// f0 (gold)
            1.5f,                           // ior
            METAL                           // type
        };

        // Top face (glass)
        materials[8] = materials[9] = {
            make_float3(1.0f),              // base_color
            0.0f,                           // metallic
            make_float3(0.0f),              // emission
            0.0f,                           // roughness
            make_float3(0.04f),             // f0
            1.5f,                           // ior
            DIELECTRIC                      // type
        };

        // Bottom face (emissive)
        materials[10] = materials[11] = {
            make_float3(1.0f),              // base_color
            0.0f,                           // metallic
            make_float3(10.0f),             // emission (bright white)
            0.0f,                           // roughness
            make_float3(0.0f),              // f0
            1.0f,                           // ior
            EMISSIVE                        // type
        };

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
        pipeline.initialize();

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
        params.frame.samples_per_pixel = 1;
        params.frame.max_bounces = 5;

        // Camera setup - position it to view the cube
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
        params.geometry.materials = reinterpret_cast<Material*>(scene.getMaterialsBuffer());
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

            // Print progress
            if ((frame + 1) % 10 == 0) {
                std::cout << "Completed " << (frame + 1) << " samples" << std::endl;
            }
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
            mapped.x = powf(pixel.x / (1.0f + pixel.x), 1.0f/2.2f);
            mapped.y = powf(pixel.y / (1.0f + pixel.y), 1.0f/2.2f);
            mapped.z = powf(pixel.z / (1.0f + pixel.z), 1.0f/2.2f);
            
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