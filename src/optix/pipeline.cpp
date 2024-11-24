#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table.h>
#include "../include/optix/pipeline.hpp"

// OptiX function table
OptixFunctionTable g_optixFunctionTable = {};

// Error check/report helper for OptiX
#define OPTIX_CHECK(call)                                                  \
    do {                                                                  \
        OptixResult res = call;                                          \
        if (res != OPTIX_SUCCESS) {                                      \
            std::stringstream ss;                                        \
            ss << "OptiX call '" << #call << "' failed: "               \
               << optixGetErrorString(res);                              \
            throw std::runtime_error(ss.str());                          \
        }                                                                \
    } while (0)

// Error check/report helper for CUDA
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                      \
            std::stringstream ss;                                        \
            ss << "CUDA call '" << #call << "' failed: "                 \
               << cudaGetErrorString(error);                             \
            throw std::runtime_error(ss.str());                          \
        }                                                                \
    } while (0)

// PathTracerPipeline implementation
PathTracerPipeline::PathTracerPipeline() : impl(std::make_unique<OptixPipelineImpl>()) {}
PathTracerPipeline::~PathTracerPipeline() = default;

void PathTracerPipeline::initialize() {
    impl->initialize();
}

void PathTracerPipeline::render(const LaunchParams& params) {
    impl->render(params);
}

OptixDeviceContext PathTracerPipeline::getContext() const {
    return impl->getContext();
}

// OptixPipelineImpl implementation
OptixPipelineImpl::OptixPipelineImpl() : context(nullptr), pipeline(nullptr), stream(nullptr) {}

OptixPipelineImpl::~OptixPipelineImpl() {
    cleanup();
}

void OptixPipelineImpl::cleanup() {
    if (pipeline) {
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = nullptr;
    }
    if (context) {
        OPTIX_CHECK(optixDeviceContextDestroy(context));
        context = nullptr;
    }
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void OptixPipelineImpl::initialize() {
    std::cout << "Creating OptiX context..." << std::endl;
    createContext();
    
    std::cout << "Loading PTX..." << std::endl;
    loadPTX();
    
    std::cout << "Creating OptiX module..." << std::endl;
    createModule();
    
    std::cout << "Creating program groups..." << std::endl;
    createProgramGroups();
    
    std::cout << "Creating pipeline..." << std::endl;
    createPipeline();
    
    std::cout << "Creating Shader Binding Table..." << std::endl;
    createSBT();

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));
}

void OptixPipelineImpl::context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    std::cout << "[" << level << "][" << tag << "]: " << message << std::endl;
}

void OptixPipelineImpl::createContext() {
    // Initialize CUDA
    std::cout << "Initializing CUDA driver API..." << std::endl;
    CUDA_CHECK(cudaFree(0));
    
    CUdevice device;
    CUresult cu_res = cuDeviceGet(&device, 0);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error getting CUDA device");
    }
    
    // Get device properties
    int compute_capability_major = 0;
    cu_res = cuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error getting compute capability major");
    }
    
    int compute_capability_minor = 0;
    cu_res = cuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error getting compute capability minor");
    }
    
    char device_name[256];
    cu_res = cuDeviceGetName(device_name, sizeof(device_name), device);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error getting device name");
    }
    
    int device_count;
    cu_res = cuDeviceGetCount(&device_count);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error getting device count");
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    std::cout << "Using CUDA device: " << device_name << std::endl;
    std::cout << "Compute capability: " << compute_capability_major << "." << compute_capability_minor << std::endl;
    
    // Create CUDA context
    CUcontext cu_ctx = nullptr;
    cu_res = cuCtxCreate(&cu_ctx, 0, device);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error creating CUDA context");
    }
    std::cout << "CUDA context created successfully" << std::endl;
    
    // Initialize OptiX
    std::cout << "Initializing OptiX function table..." << std::endl;
    OPTIX_CHECK(optixInit());
    
    // Create OptiX device context
    std::cout << "Creating OptiX device context..." << std::endl;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
    std::cout << "OptiX context created successfully" << std::endl;
}

void OptixPipelineImpl::loadPTX() {
    const std::string ptx_filename = "cmake-build-release/ptx/optix_shaders.ptx";
    std::ifstream ptx_file(ptx_filename, std::ios::binary);
    if (!ptx_file) {
        throw std::runtime_error("Failed to open PTX file: " + ptx_filename);
    }
    
    ptx_file.seekg(0, std::ios::end);
    size_t size = ptx_file.tellg();
    ptx_file.seekg(0, std::ios::beg);
    
    ptx_code.resize(size);
    ptx_file.read(ptx_code.data(), size);
    
    std::cout << "PTX loaded successfully (" << size << " bytes)" << std::endl;
}

void OptixPipelineImpl::createModule() {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    
    pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;  // Simplified payload system
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code.data(),
        ptx_code.size(),
        log,
        &sizeof_log,
        &module
    ));
    
    if (sizeof_log > 1) {
        std::cout << "Module compile log: " << log << std::endl;
    }
}

void OptixPipelineImpl::createProgramGroups() {
    OptixProgramGroupOptions program_group_options = {};
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    // Ray generation program group
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &raygen_prog_group_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        &raygen_prog_group
    ));
    
    // Miss program groups
    OptixProgramGroupDesc miss_prog_group_desc[2] = {};
    miss_prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc[0].miss.module = module;
    miss_prog_group_desc[0].miss.entryFunctionName = "__miss__radiance";
    miss_prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc[1].miss.module = module;
    miss_prog_group_desc[1].miss.entryFunctionName = "__miss__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        miss_prog_group_desc,
        2,
        &program_group_options,
        log,
        &sizeof_log,
        miss_prog_groups
    ));
    
    // Hit group program groups
    OptixProgramGroupDesc hitgroup_prog_group_desc[2] = {};
    hitgroup_prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc[0].hitgroup.moduleCH = module;
    hitgroup_prog_group_desc[0].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitgroup_prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc[1].hitgroup.moduleCH = module;
    hitgroup_prog_group_desc[1].hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        hitgroup_prog_group_desc,
        2,
        &program_group_options,
        log,
        &sizeof_log,
        hitgroup_prog_groups
    ));
    
    std::cout << "Program groups created successfully" << std::endl;
}

void OptixPipelineImpl::createPipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_prog_group,
        miss_prog_groups[0],
        miss_prog_groups[1],
        hitgroup_prog_groups[0],
        hitgroup_prog_groups[1]
    };
    
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 5;  // Allow for multiple bounces
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline
    ));
    
    if (sizeof_log > 1) {
        std::cout << "Pipeline link log: " << log << std::endl;
    }
    
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        2048,  // direct stack size
        2048,  // continuation stack size
        2048,  // maximum traversable graph depth
        1      // maximum number of traversables in a single trace
    ));
    
    std::cout << "Pipeline created successfully" << std::endl;
}

void OptixPipelineImpl::createSBT() {
    // Create raygen records
    RayGenSbtRecord raygen_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &raygen_record));
    void* d_raygen_record_ptr;
    CUDA_CHECK(cudaMalloc(&d_raygen_record_ptr, sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(
        d_raygen_record_ptr,
        &raygen_record,
        sizeof(RayGenSbtRecord),
        cudaMemcpyHostToDevice
    ));
    d_raygen_record = reinterpret_cast<CUdeviceptr>(d_raygen_record_ptr);
    
    // Create miss records
    MissSbtRecord miss_records[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[0], &miss_records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[1], &miss_records[1]));
    void* d_miss_record_ptr;
    CUDA_CHECK(cudaMalloc(&d_miss_record_ptr, sizeof(MissSbtRecord) * 2));
    CUDA_CHECK(cudaMemcpy(
        d_miss_record_ptr,
        miss_records,
        sizeof(MissSbtRecord) * 2,
        cudaMemcpyHostToDevice
    ));
    d_miss_record = reinterpret_cast<CUdeviceptr>(d_miss_record_ptr);
    
    // Create hitgroup records
    HitGroupSbtRecord hitgroup_records[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[0], &hitgroup_records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[1], &hitgroup_records[1]));
    void* d_hitgroup_record_ptr;
    CUDA_CHECK(cudaMalloc(&d_hitgroup_record_ptr, sizeof(HitGroupSbtRecord) * 2));
    CUDA_CHECK(cudaMemcpy(
        d_hitgroup_record_ptr,
        hitgroup_records,
        sizeof(HitGroupSbtRecord) * 2,
        cudaMemcpyHostToDevice
    ));
    d_hitgroup_record = reinterpret_cast<CUdeviceptr>(d_hitgroup_record_ptr);
    
    std::cout << "Shader Binding Table created successfully" << std::endl;
}

void OptixPipelineImpl::render(const LaunchParams& params) {
    std::cout << "Starting render..." << std::endl;
    
    void* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(
        d_params,
        &params,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice
    ));
    
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 2;
    sbt.hitgroupRecordBase = d_hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 2;
    
    OPTIX_CHECK(optixLaunch(
        pipeline,
        stream,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(LaunchParams),
        &sbt,
        params.frame.width,   // Launch width
        params.frame.height,  // Launch height
        1                    // Launch depth
    ));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_params));
    
    std::cout << "Render completed" << std::endl;
}

OptixDeviceContext OptixPipelineImpl::getContext() const {
    return context;
}
