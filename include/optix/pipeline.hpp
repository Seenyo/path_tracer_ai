#pragma once

#include <memory>
#include <vector>
#include <optix.h>
#include "launch_params.hpp"

// Forward declarations
class OptixPipelineImpl;

/**
 * @brief High-level interface for the Path Tracer Pipeline.
 *
 * Manages the initialization and rendering processes by delegating
 * tasks to the underlying OptixPipelineImpl.
 */
class PathTracerPipeline {
public:
    PathTracerPipeline();
    ~PathTracerPipeline();

    /**
     * @brief Initializes the OptiX pipeline with the given materials.
     *
     * @param materials A vector of materials used in the scene.
     */
    void initialize(const std::vector<Material>& materials);

    /**
     * @brief Renders the scene using the provided launch parameters.
     *
     * @param params The launch parameters containing camera, frame, and scene data.
     */
    void render(const LaunchParams& params);

    /**
     * @brief Retrieves the OptiX device context.
     *
     * @return OptixDeviceContext The OptiX device context.
     */
    OptixDeviceContext getContext() const;

private:
    std::unique_ptr<OptixPipelineImpl> impl; /**< Implementation details hidden via Pimpl idiom */
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
    void allocatePayloadBuffer(unsigned int width, unsigned int height);
    void cleanup();

    static void context_log_cb(unsigned int level, const char* tag, const char* message, void*);

    OptixDeviceContext context{0};
    OptixModule module{0};
    OptixPipeline pipeline{0};
    OptixPipelineCompileOptions pipeline_compile_options{};

    OptixProgramGroup raygen_prog_group{nullptr};
    OptixProgramGroup miss_prog_groups[2]{nullptr, nullptr};
    OptixProgramGroup hitgroup_prog_groups[2]{nullptr, nullptr};

    CUdeviceptr d_raygen_record{0};
    CUdeviceptr d_miss_record{0};
    CUdeviceptr d_hitgroup_record{0};

    CUstream stream{0};
    std::vector<char> ptx_code;
    std::vector<Material> materials;
    OptixShaderBindingTable sbt = {};

    Payload* payload_buffer{nullptr};

};

