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
#include "../../include/gpu/cuda_utils.hpp" // For CUDABuffer and error checking macros
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

#define CU_CHECK(call)                                                    \
    do {                                                                  \
        CUresult error = call;                                            \
        if (error != CUDA_SUCCESS) {                                      \
            const char *errName = nullptr;                                \
            const char *errString = nullptr;                              \
            cuGetErrorName(error, &errName);                              \
            cuGetErrorString(error, &errString);                          \
            std::stringstream ss;                                         \
            ss << "CUDA Driver API call (" << #call << ") failed with error: " \
               << errName << " (" << errString << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str());                           \
        }                                                                 \
    } while (0)

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
    CU_CHECK(cuCtxGetCurrent(&cuCtx));

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
    pipelineCompileOptions.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);


    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreate(
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
}

void OptixRenderer::buildAccelerationStructure(const std::vector<Triangle>& triangles) {
    // Convert triangles to OptiX-compatible format
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<int> materialIds;

    vertices.reserve(triangles.size() * 3);
    indices.reserve(triangles.size());
    normals.reserve(triangles.size() * 3);
    texcoords.reserve(triangles.size() * 3);
    materialIds.reserve(triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        
        // Add vertices
        vertices.push_back(glmToCuda(tri.v0));
        vertices.push_back(glmToCuda(tri.v1));
        vertices.push_back(glmToCuda(tri.v2));
        
        // Add indices
        indices.push_back(make_uint3(i*3, i*3+1, i*3+2));
        
        // Add normals
        normals.push_back(glmToCuda(tri.n0));
        normals.push_back(glmToCuda(tri.n1));
        normals.push_back(glmToCuda(tri.n2));
        
        // Add texture coordinates
        texcoords.push_back(make_float2(tri.uv0.x, tri.uv0.y));
        texcoords.push_back(make_float2(tri.uv1.x, tri.uv1.y));
        texcoords.push_back(make_float2(tri.uv2.x, tri.uv2.y));
        
        // Add material ID
        materialIds.push_back(tri.materialId);
    }

    // Create vertex buffer
    // Use CUDA Driver API functions
    CUdeviceptr d_vertices;
    CU_CHECK(cuMemAlloc(&d_vertices, vertices.size() * sizeof(float3)));
    CU_CHECK(cuMemcpyHtoD(d_vertices, vertices.data(), vertices.size() * sizeof(float3)));

    CUdeviceptr d_indices;
    CU_CHECK(cuMemAlloc(&d_indices, indices.size() * sizeof(uint3)));
    CU_CHECK(cuMemcpyHtoD(d_indices, indices.data(), indices.size() * sizeof(uint3)));

    // Build input for acceleration structure
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Set up triangle array
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    buildInput.triangleArray.vertexBuffers = &d_vertices;

    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    buildInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
    buildInput.triangleArray.indexBuffer = d_indices;

    // Set up flags
    unsigned int buildFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    buildInput.triangleArray.flags = buildFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    // Set up build options
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Get size requirements
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &buildInput,
        1,  // num build inputs
        &gasBufferSizes
    ));

    // Allocate buffers
    CUdeviceptr d_temp;
    CU_CHECK(cuMemAlloc(&d_temp, gasBufferSizes.tempSizeInBytes));
    CUdeviceptr d_gas;
    CU_CHECK(cuMemAlloc(&d_gas, gasBufferSizes.outputSizeInBytes));

    // Build GAS
    OPTIX_CHECK(optixAccelBuild(
        context,
        0,  // CUDA stream
        &accelOptions,
        &buildInput,
        1,  // num build inputs
        d_temp,
        gasBufferSizes.tempSizeInBytes,
        d_gas,
        gasBufferSizes.outputSizeInBytes,
        &gasHandle,
        nullptr,  // emitted property list
        0         // num emitted properties
    ));

    // Wait for build to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up temporary buffers
    CU_CHECK(cuMemFree(d_temp));
    CU_CHECK(cuMemFree(d_vertices));
    CU_CHECK(cuMemFree(d_indices));


    // Store the GAS buffer in our class member
    this->d_gas = CUDABuffer<unsigned char>();
    this->d_gas.alloc(gasBufferSizes.outputSizeInBytes);
    CUDA_CHECK(cudaMemcpy(this->d_gas.get(), reinterpret_cast<void*>(d_gas), gasBufferSizes.outputSizeInBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas)));
}

void OptixRenderer::setupLaunchParams(const Camera& camera) {
    // Convert camera to GPU format
    GPUCamera gpuCamera;
    gpuCamera.position = glmToCuda(camera.getPosition());
    gpuCamera.forward = glmToCuda(camera.getForward());
    gpuCamera.right = glmToCuda(camera.getRight());
    gpuCamera.up = glmToCuda(camera.getUp());
    gpuCamera.fov = camera.getFOV();
    gpuCamera.aspectRatio = static_cast<float>(settings.width) / settings.height;

    // Update launch parameters
    launchParams.width = settings.width;
    launchParams.height = settings.height;
    launchParams.samplesPerPixel = settings.samplesPerPixel;
    launchParams.maxBounces = settings.maxBounces;
    launchParams.gamma = settings.gamma;
    launchParams.camera = gpuCamera;
    launchParams.colorBuffer = d_colorBuffer.get();
    launchParams.seeds = d_seeds.get();
    launchParams.traversable = gasHandle;

    // Upload launch parameters to GPU
    if (!d_launchParams.get()) {
        d_launchParams.alloc(1);
    }
    d_launchParams.upload(&launchParams, 1);
}

void OptixRenderer::uploadScene(const Scene& scene) {
    // Convert scene data to GPU format
    std::vector<Triangle> triangles = scene.getTriangles();
    buildAccelerationStructure(triangles);

    // Convert materials
    std::vector<GPUMaterial> gpuMaterials = convertMaterials(scene.getMaterials());
    d_materials.alloc_and_upload(gpuMaterials);

    // Convert lights
    std::vector<GPULight> gpuLights = convertLights(scene.getLights());
    d_lights.alloc_and_upload(gpuLights);

    // Initialize color buffer
    size_t pixelCount = settings.width * settings.height;
    d_colorBuffer.alloc(pixelCount);
    d_seeds.alloc(pixelCount);

    // Initialize random seeds
    initializeRNG();

    // Set material and light pointers in launch parameters
    launchParams.materials = d_materials.get();
    launchParams.numMaterials = static_cast<unsigned int>(gpuMaterials.size());
    launchParams.lights = d_lights.get();
    launchParams.numLights = static_cast<unsigned int>(gpuLights.size());
}

void OptixRenderer::initializeRNG() {
    size_t pixelCount = settings.width * settings.height;
    std::vector<unsigned int> seeds(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) {
        seeds[i] = static_cast<unsigned int>(rand());
    }
    d_seeds.upload(seeds.data(), pixelCount);
}

void OptixRenderer::render(const Camera& camera) {
    if (!isInitialized) {
        throw std::runtime_error("Renderer not initialized. Call initialize() before rendering.");
    }

    // Update launch parameters
    setupLaunchParams(camera);

    // Set up shader binding table (SBT)
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = createSBTRecord(raygenPG);
    sbt.missRecordBase = createSBTRecord(missPG);
    sbt.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = createSBTRecord(hitgroupPG);
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt.hitgroupRecordCount = 1;

    // Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(
        pipeline,
        0, // CUDA stream
        reinterpret_cast<CUdeviceptr>(d_launchParams.get()),
        sizeof(LaunchParams),
        &sbt,
        settings.width,
        settings.height,
        1 // Depth
    ));

    // Wait for the GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up SBT records
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
}

void OptixRenderer::saveImage(const std::string& filename) {
    size_t pixelCount = settings.width * settings.height;
    std::vector<float4> hostPixels(pixelCount);
    d_colorBuffer.download(hostPixels.data(), pixelCount);

    // Apply gamma correction and convert to 8-bit RGB
    std::vector<unsigned char> imageData(pixelCount * 3);
    for (size_t i = 0; i < pixelCount; ++i) {
        float3 color = make_float3(hostPixels[i].x, hostPixels[i].y, hostPixels[i].z);
        color = powf(color, 1.0f / settings.gamma);
        imageData[i * 3 + 0] = static_cast<unsigned char>(clamp(color.x, 0.0f, 1.0f) * 255.0f);
        imageData[i * 3 + 1] = static_cast<unsigned char>(clamp(color.y, 0.0f, 1.0f) * 255.0f);
        imageData[i * 3 + 2] = static_cast<unsigned char>(clamp(color.z, 0.0f, 1.0f) * 255.0f);
    }

    // Save image using your preferred image library (e.g., stb_image_write)
    // Example with stb_image_write:
    // stbi_write_png(filename.c_str(), settings.width, settings.height, 3, imageData.data(), settings.width * 3);

    // For this example, we'll just print a message
    std::cout << "Image saved as: " << filename << std::endl;
}

void OptixRenderer::cleanup() {
    if (pipeline) {
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = nullptr;
    }
    if (module) {
        OPTIX_CHECK(optixModuleDestroy(module));
        module = nullptr;
    }
    if (raygenPG) {
        OPTIX_CHECK(optixProgramGroupDestroy(raygenPG));
        raygenPG = nullptr;
    }
    if (missPG) {
        OPTIX_CHECK(optixProgramGroupDestroy(missPG));
        missPG = nullptr;
    }
    if (hitgroupPG) {
        OPTIX_CHECK(optixProgramGroupDestroy(hitgroupPG));
        hitgroupPG = nullptr;
    }
    if (context) {
        OPTIX_CHECK(optixDeviceContextDestroy(context));
        context = nullptr;
    }

    // Free device buffers
    d_materials.free();
    d_lights.free();
    d_colorBuffer.free();
    d_seeds.free();
    d_launchParams.free();
    d_gas.free();
}

std::vector<GPUMaterial> OptixRenderer::convertMaterials(const std::vector<std::shared_ptr<Material>>& materials) {
    std::vector<GPUMaterial> gpuMaterials;
    for (const auto& mat : materials) {
        GPUMaterial gpuMat = {};
        gpuMat.type = static_cast<int>(mat->type);
        gpuMat.albedo = make_float3(mat->albedo.r, mat->albedo.g, mat->albedo.b);
        gpuMat.roughness = mat->roughness;
        gpuMat.metallic = mat->metallic;
        gpuMat.ior = mat->ior;
        gpuMaterials.push_back(gpuMat);
    }
    return gpuMaterials;
}

std::vector<GPULight> OptixRenderer::convertLights(const std::vector<Light>& lights) {
    std::vector<GPULight> gpuLights;
    for (const auto& light : lights) {
        GPULight gpuLight = {};
        gpuLight.position = make_float3(light.position.x, light.position.y, light.position.z);
        gpuLight.color = make_float3(light.color.r, light.color.g, light.color.b);
        gpuLight.intensity = light.intensity;
        gpuLights.push_back(gpuLight);
    }
    return gpuLights;
}

CUdeviceptr OptixRenderer::createSBTRecord(OptixProgramGroup& programGroup) {
    SbtRecord sbtRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, &sbtRecord));

    CUdeviceptr d_sbtRecord;
    CU_CHECK(cuMemAlloc(&d_sbtRecord, sizeof(SbtRecord)));
    CU_CHECK(cuMemcpyHtoD(d_sbtRecord, &sbtRecord, sizeof(SbtRecord)));

    return d_sbtRecord;
}
