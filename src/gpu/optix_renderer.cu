#include "../../include/gpu/optix_renderer.hpp"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <fstream>

// Include your utility headers
#include "cuda_utils.cu" // For CUDABuffer and error checking macros
#include "../../include/gpu/optix_types.hpp"
#include "../../include/camera.hpp"
#include "../../include/scene.hpp"

// OptiX function table
static OptixFunctionTable optixFunctionTable;

// Path to the PTX file generated from optix_kernels.cu
#define OPTIX_PTX_PATH "src/gpu/ptx/optix_kernels.ptx"

// OptiX constants and macros
#ifndef OPTIX_CHECK
#define OPTIX_CHECK( call )                                                                          \
    do {                                                                                            \
        OptixResult res = call;                                                                     \
        if (res != OPTIX_SUCCESS) {                                                                 \
            std::stringstream ss;                                                                   \
            ss << "OptiX call (" << #call << ") failed with error: "                               \
               << optixGetErrorName(res) << " (" << res << ")";                                    \
            throw std::runtime_error(ss.str());                                                     \
        }                                                                                           \
    } while (0)
#endif

#ifndef OPTIX_CHECK_LOG
#define OPTIX_CHECK_LOG(call)                                                                       \
    do {                                                                                            \
        OptixResult res = call;                                                                     \
        if (res != OPTIX_SUCCESS) {                                                                 \
            std::stringstream ss;                                                                   \
            ss << "OptiX call (" << #call << ") failed with error: "                               \
               << optixGetErrorName(res) << " (" << res << ")";                                    \
            throw std::runtime_error(ss.str());                                                     \
        }                                                                                           \
    } while (0)
#endif

// Helper functions
__host__ __device__ float clamp(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__host__ __device__ float3 powf(const float3& base, float exponent) {
    return make_float3(
        powf(base.x, exponent),
        powf(base.y, exponent),
        powf(base.z, exponent)
    );
}

// Definition of SBT record structure (must be defined before use)
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // You can add custom data here if needed
};

OptixRenderer::OptixRenderer(const Settings& settings)
    : settings(settings), isInitialized(false), context(nullptr) {
    // Initialize member variables if needed
}

OptixRenderer::~OptixRenderer() {
    cleanup();
}

void OptixRenderer::initialize() {
    // Initialize OptiX context and pipeline
    createContext();
    createModule();
    createProgramGroups();
    createPipeline();
    isInitialized = true;
}

void OptixRenderer::createContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    CUcontext cuCtx = nullptr;
    CUDA_CHECK(cuCtxGetCurrent(&cuCtx));

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;

    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
}

void OptixRenderer::createModule() {
    // Read PTX code from file
    std::ifstream ptxFile(OPTIX_PTX_PATH);
    if (!ptxFile.is_open()) {
        throw std::runtime_error("Failed to open PTX file: " OPTIX_PTX_PATH);
    }
    std::string ptxCode((std::istreambuf_iterator<char>(ptxFile)),
                        std::istreambuf_iterator<char>());
    ptxFile.close();

    // Module compilation options
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    // Pipeline compilation options
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 3;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log,
        &logSize,
        &module));
}

void OptixRenderer::createProgramGroups() {
    char log[2048];
    size_t logSize = sizeof(log);

    // Ray generation program group
    OptixProgramGroupOptions programGroupOptions = {};
    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenPGDesc.raygen.module = module;
    raygenPGDesc.raygen.entryFunctionName = "__raygen__rg";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context,
        &raygenPGDesc,
        1,
        &programGroupOptions,
        log,
        &logSize,
        &raygenPG));

    // Miss program group
    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = module;
    missPGDesc.miss.entryFunctionName = "__miss__ms";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context,
        &missPGDesc,
        1,
        &programGroupOptions,
        log,
        &logSize,
        &missPG));

    // Hit group program group
    OptixProgramGroupDesc hitgroupPGDesc = {};
    hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupPGDesc.hitgroup.moduleCH = module;
    hitgroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroupPGDesc.hitgroup.moduleAH = module;
    hitgroupPGDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context,
        &hitgroupPGDesc,
        1,
        &programGroupOptions,
        log,
        &logSize,
        &hitgroupPG));
}

void OptixRenderer::createPipeline() {
    OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = settings.maxBounces;

    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context,
        nullptr, // Pipeline compile options (already set during module creation)
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
        log,
        &logSize,
        &pipeline));

    // Stack sizes
    OptixStackSizes stackSizes = {};
    for (auto& progGroup : programGroups) {
        OptixStackSizes groupSizes = {};
        OPTIX_CHECK(optixProgramGroupGetStackSize(progGroup, &groupSizes));
        stackSizes.cssRG = std::max(stackSizes.cssRG, groupSizes.cssRG);
        stackSizes.cssMS = std::max(stackSizes.cssMS, groupSizes.cssMS);
        stackSizes.cssCH = std::max(stackSizes.cssCH, groupSizes.cssCH);
        stackSizes.cssAH = std::max(stackSizes.cssAH, groupSizes.cssAH);
        stackSizes.cssIS = std::max(stackSizes.cssIS, groupSizes.cssIS);
        stackSizes.cssCC = std::max(stackSizes.cssCC, groupSizes.cssCC);
        stackSizes.dssDC = std::max(stackSizes.dssDC, groupSizes.dssDC);
    }

    uint32_t maxTraceDepth = settings.maxBounces;
    uint32_t maxCCDepth = 0;  // No continuation callables
    uint32_t maxDCDepth = 0;  // No direct callables

    uint32_t directCallableStackSizeFromTraversal = stackSizes.dssDC * maxDCDepth;
    uint32_t directCallableStackSizeFromState = stackSizes.dssDC * maxDCDepth;
    uint32_t continuationStackSize = stackSizes.cssRG + stackSizes.cssMS + stackSizes.cssCH +
                                   stackSizes.cssAH + stackSizes.cssIS + stackSizes.cssCC * maxCCDepth;
    uint32_t maxTraversableGraphDepth = 1;

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        maxTraversableGraphDepth));
}

// ... Rest of the implementation remains the same ...
