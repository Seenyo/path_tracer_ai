#pragma once

#include <memory>
#include <string>
#include <vector>
#include "optix_types.hpp"
#include "cuda_utils.hpp"
#include "../scene.hpp"
#include "../camera.hpp"

class OptixRenderer
{
public:
    struct Settings
    {
        int width;
        int height;
        int samplesPerPixel;
        int maxBounces;
        float gamma;

        Settings()
            : width(800), height(450), samplesPerPixel(10), maxBounces(3), gamma(2.2f) {}
    };

    OptixRenderer(const Settings &settings = Settings());
    ~OptixRenderer();

    // Initialize OptiX context and pipeline
    void initialize();

    // Upload scene data to GPU
    void uploadScene(const Scene &scene);

    // Set launch parameters
    void setLaunchParams(const LaunchParams &params);

    // Render frame
    void render(const Camera &camera);

    // Save rendered image
    void saveImage(const std::string &filename);

private:
    // OptiX context and pipeline
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixModule module;
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPG;
    OptixProgramGroup hitgroupPG;

    // Launch parameters
    LaunchParams launchParams;
    CUDABuffer<LaunchParams> d_launchParams;

    // Scene data buffers
    CUDABuffer<GPUMaterial> d_materials;
    CUDABuffer<GPULight> d_lights;
    CUDABuffer<float4> d_colorBuffer;
    CUDABuffer<unsigned int> d_seeds;

    // Acceleration structure
    OptixTraversableHandle gasHandle;
    CUDABuffer<unsigned char> d_gas;
    CUDABuffer<OptixInstance> d_instances;
    CUDABuffer<unsigned char> d_ias;

    // Settings
    Settings settings;
    bool isInitialized;

    // Helper functions
    void createContext();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildAccelerationStructure(const std::vector<Triangle> &triangles);
    void setupLaunchParams(const Camera &camera);
    void initializeRNG();

    // Convert scene data to GPU format
    std::vector<GPUMaterial> convertMaterials(const std::vector<std::shared_ptr<Material>> &materials);
    std::vector<GPULight> convertLights(const std::vector<Light> &lights);

    // Clean up resources
    void cleanup();
};
