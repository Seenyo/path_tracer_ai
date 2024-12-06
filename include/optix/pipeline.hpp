// pipeline.hpp

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

/**
 * @brief Implementation class for the OptiX pipeline.
 *
 * Handles the detailed setup and execution of the OptiX rendering pipeline,
 * including context creation, module loading, program group creation,
 * pipeline configuration, Shader Binding Table (SBT) setup, and payload management.
 */
class OptixPipelineImpl {
public:
    /**
     * @brief Constructs the OptixPipelineImpl with the provided scene materials.
     *
     * @param scene_materials A vector of materials used in the scene.
     */
    OptixPipelineImpl(const std::vector<Material>& scene_materials);

    /**
     * @brief Destructs the OptixPipelineImpl, ensuring proper cleanup.
     */
    ~OptixPipelineImpl();

    /**
     * @brief Initializes the OptiX pipeline, including context, modules, program groups, pipeline, SBT, and payload buffer.
     */
    void initialize();

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
    // Core pipeline setup functions
    void createContext();
    void loadPTX();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();
    void allocatePayloadBuffer(unsigned int width, unsigned int height);
    void cleanup();

    // Logging callback for OptiX
    static void context_log_cb(unsigned int level, const char* tag, const char* message, void*);

    // OptiX objects
    OptixDeviceContext context{0}; /**< OptiX device context */
    OptixModule module{0}; /**< OptiX module */
    OptixPipeline pipeline{0}; /**< OptiX pipeline */
    OptixPipelineCompileOptions pipeline_compile_options{}; /**< Compilation options for the pipeline */

    // Program groups for different ray types
    OptixProgramGroup raygen_prog_group{nullptr}; /**< Ray generation program group */
    OptixProgramGroup miss_prog_groups[2]{nullptr, nullptr};     /**< Miss program groups: [0] = radiance, [1] = shadow */
    OptixProgramGroup hitgroup_prog_groups[2]{nullptr, nullptr}; /**< Hit group program groups: [0] = radiance, [1] = shadow */

    // Device memory for SBT records
    CUdeviceptr d_raygen_record{0}; /**< Device pointer to ray generation SBT record */
    CUdeviceptr d_miss_record{0}; /**< Device pointer to miss SBT records */
    CUdeviceptr d_hitgroup_record{0}; /**< Device pointer to hit group SBT records */

    // CUDA stream
    CUstream stream{0}; /**< CUDA stream for asynchronous operations */

    // PTX code storage
    std::vector<char> ptx_code; /**< Buffer to store PTX code */

    // Materials
    std::vector<Material> materials; /**< Vector of materials used in the scene */

    // Shader Binding Table
    OptixShaderBindingTable sbt = {}; /**< Shader Binding Table containing all program groups */

    // Payload buffer
    Payload* payload_buffer{nullptr}; /**< Device pointer to the payload buffer */
};
