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
#include <optix_function_table_definition.h>
#include "../include/optix/pipeline.hpp"

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
PathTracerPipeline::PathTracerPipeline() : impl(nullptr) {}
PathTracerPipeline::~PathTracerPipeline() = default;

void PathTracerPipeline::initialize(const std::vector<Material>& materials) {
    impl = std::make_unique<OptixPipelineImpl>(materials);
    impl->initialize();
}

void PathTracerPipeline::render(const LaunchParams& params) {
    impl->render(params);
}

OptixDeviceContext PathTracerPipeline::getContext() const {
    return impl->getContext();
}

// OptixPipelineImpl implementation
OptixPipelineImpl::OptixPipelineImpl(const std::vector<Material>& scene_materials)
        : materials(scene_materials), context(nullptr), pipeline(nullptr), stream(nullptr) {}

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
    const std::string ptx_filename = "../cmake-build-release/ptx/optix_shaders.ptx";
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
    pipeline_compile_options.numPayloadValues = 2;  // Reduced since we're using payload pointers
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

    std::cout << "Pipeline compile options set with numPayloadValues = "
              << pipeline_compile_options.numPayloadValues << std::endl;

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

    std::cout << "OptiX module created successfully." << std::endl;
}

// ... Other includes and code ...

void OptixPipelineImpl::createProgramGroups() {
    // Program group for ray generation
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

    // Program groups for miss shaders
    OptixProgramGroupDesc miss_prog_group_descs[2] = {};

    // Radiance miss program
    miss_prog_group_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_descs[0].miss.module = module;
    miss_prog_group_descs[0].miss.entryFunctionName = "__miss__radiance";

    // Shadow miss program
    miss_prog_group_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_descs[1].miss.module = module;
    miss_prog_group_descs[1].miss.entryFunctionName = "__miss__shadow";

    // Program groups for hit groups
    OptixProgramGroupDesc hitgroup_prog_group_descs[2] = {};

    // Radiance hit group
    hitgroup_prog_group_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_descs[0].hitgroup.moduleCH = module;
    hitgroup_prog_group_descs[0].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitgroup_prog_group_descs[0].hitgroup.moduleAH = nullptr;
    hitgroup_prog_group_descs[0].hitgroup.entryFunctionNameAH = nullptr;
    hitgroup_prog_group_descs[0].hitgroup.moduleIS = nullptr;
    hitgroup_prog_group_descs[0].hitgroup.entryFunctionNameIS = nullptr;

    // Shadow hit group (no closest hit program)
    hitgroup_prog_group_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_descs[1].hitgroup.moduleCH = module;
    hitgroup_prog_group_descs[1].hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitgroup_prog_group_descs[1].hitgroup.moduleAH = nullptr;
    hitgroup_prog_group_descs[1].hitgroup.entryFunctionNameAH = nullptr;
    hitgroup_prog_group_descs[1].hitgroup.moduleIS = nullptr;
    hitgroup_prog_group_descs[1].hitgroup.entryFunctionNameIS = nullptr;

    // Program group options
    OptixProgramGroupOptions program_group_options = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Create raygen program group
    OPTIX_CHECK(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group
    ));

    // Create miss program groups
    OPTIX_CHECK(optixProgramGroupCreate(
            context,
            miss_prog_group_descs,
            2,
            &program_group_options,
            log,
            &sizeof_log,
            miss_prog_groups
    ));

    // Create hitgroup program groups
    OPTIX_CHECK(optixProgramGroupCreate(
            context,
            hitgroup_prog_group_descs,
            2,
            &program_group_options,
            log,
            &sizeof_log,
            hitgroup_prog_groups
    ));
}

void OptixPipelineImpl::createSBT() {
    // Create raygen records
    RayGenSbtRecord raygen_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &raygen_record));
    // No additional data for raygen record

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    // Create miss records
    MissSbtRecord miss_records[2] = {};
    // Radiance miss record
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[0], &miss_records[0]));
    miss_records[0].data.bg_color = make_float3(0.0f, 0.0f, 0.0f); // You can set your desired background color here

    // Shadow miss record
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[1], &miss_records[1]));
    // No additional data for shadow miss record

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissSbtRecord) * 2));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), miss_records, sizeof(MissSbtRecord) * 2, cudaMemcpyHostToDevice));

    // Create hitgroup records
    size_t num_materials = materials.size();
    std::vector<HitGroupSbtRecord> hitgroup_records(num_materials * RAY_TYPE_COUNT);

    for (size_t i = 0; i < num_materials; ++i) {
        // Radiance hit group record
        HitGroupSbtRecord& radiance_record = hitgroup_records[i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[RAY_TYPE_RADIANCE], &radiance_record));
        radiance_record.material = materials[i];

        // Shadow hit group record
        HitGroupSbtRecord& shadow_record = hitgroup_records[i * RAY_TYPE_COUNT + RAY_TYPE_SHADOW];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[RAY_TYPE_SHADOW], &shadow_record));
        // No material data needed for shadow rays
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof(HitGroupSbtRecord) * hitgroup_records.size()));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), hitgroup_records.data(), sizeof(HitGroupSbtRecord) * hitgroup_records.size(), cudaMemcpyHostToDevice));

    // Initialize SBT
    sbt = {};
    sbt.raygenRecord = d_raygen_record;

    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 2;

    sbt.hitgroupRecordBase = d_hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
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

//void OptixPipelineImpl::createSBT() {
//    // Create raygen records
//    RayGenSbtRecord raygen_record = {};
//    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &raygen_record));
//    // No additional data for raygen record
//
//    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenSbtRecord)));
//    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));
//
//    // Create miss records
//    MissSbtRecord miss_records[2] = {};
//    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[0], &miss_records[0]));
//    // Set miss data if needed (e.g., background color)
//
//    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[1], &miss_records[1]));
//    // Set miss data for shadow rays if needed
//
//    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissSbtRecord) * 2));
//    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), miss_records, sizeof(MissSbtRecord) * 2, cudaMemcpyHostToDevice));
//
//    // Create hitgroup records
//    size_t num_materials = materials.size();
//    std::vector<HitGroupSbtRecord> hitgroup_records(num_materials);
//
//    for (size_t i = 0; i < num_materials; ++i) {
//        HitGroupSbtRecord& record = hitgroup_records[i];
//        // Use the appropriate program group
//        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[0], &record));
//        // Assign the material to the SBT record
//        record.material = materials[i];
//    }
//
//    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof(HitGroupSbtRecord) * num_materials));
//    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), hitgroup_records.data(), sizeof(HitGroupSbtRecord) * num_materials, cudaMemcpyHostToDevice));
//
//    // Initialize SBT
//    sbt = {};
//    sbt.raygenRecord = d_raygen_record;
//    sbt.missRecordBase = d_miss_record;
//    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
//    sbt.missRecordCount = 2;
//    sbt.hitgroupRecordBase = d_hitgroup_record;
//    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
//    sbt.hitgroupRecordCount = static_cast<unsigned int>(num_materials);
//
//    std::cout << "Shader Binding Table created successfully" << std::endl;
//}

void OptixPipelineImpl::render(const LaunchParams& params) {
    std::cout << "Starting render..." << std::endl;

    void* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
            pipeline,
            stream,
            reinterpret_cast<CUdeviceptr>(d_params),
            sizeof(LaunchParams),
            &sbt,
            params.frame.width,   // Launch width
            params.frame.height,  // Launch height
            1                     // Launch depth
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_params));

    std::cout << "Render completed" << std::endl;
}

OptixDeviceContext OptixPipelineImpl::getContext() const {
    return context;
}
