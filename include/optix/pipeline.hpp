#pragma once

#include <memory>
#include <vector>
#include <optix.h>
#include "launch_params.hpp"

// Forward declarations
class OptixPipelineImpl;

class PathTracerPipeline {
public:
    PathTracerPipeline();
    ~PathTracerPipeline();

    void initialize(const std::vector<Material>& materials);
    void render(const LaunchParams& params);
    OptixDeviceContext getContext() const;

private:
    std::unique_ptr<OptixPipelineImpl> impl;
};

class OptixPipelineImpl {
public:
    OptixPipelineImpl(const std::vector<Material>& scene_materials);
    ~OptixPipelineImpl();

    void initialize();
    void render(const LaunchParams& params);
    OptixDeviceContext getContext() const;

private:
    void createContext();
    void loadPTX();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();
    void cleanup();

    static void context_log_cb(unsigned int level, const char* tag, const char* message, void*);

    // OptiX objects
    OptixDeviceContext context{nullptr};
    OptixModule module{nullptr};
    OptixPipeline pipeline{nullptr};
    OptixPipelineCompileOptions pipeline_compile_options{};

    // Program groups for different ray types
    OptixProgramGroup raygen_prog_group{nullptr};
    OptixProgramGroup miss_prog_groups[2]{nullptr, nullptr};     // [0] = radiance, [1] = shadow
    OptixProgramGroup hitgroup_prog_groups[2]{nullptr, nullptr}; // [0] = radiance, [1] = shadow

    // Device memory for SBT records
    CUdeviceptr d_raygen_record{0};
    CUdeviceptr d_miss_record{0};
    CUdeviceptr d_hitgroup_record{0};

    // CUDA stream
    CUstream stream{0};

    // PTX code storage
    std::vector<char> ptx_code;

    // Materials
    std::vector<Material> materials;

    // Shader Binding Table
    OptixShaderBindingTable sbt = {};
};
