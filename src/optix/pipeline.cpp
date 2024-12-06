// pipeline.cpp

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
        cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                       \
            std::stringstream ss;                                        \
            ss << "CUDA call '" << #call << "' failed: "                 \
               << cudaGetErrorString(error) << std::endl;                \
            throw std::runtime_error(ss.str());                          \
        }                                                                 \
    } while (0)

// OptixPipelineImpl implementation
OptixPipelineImpl::OptixPipelineImpl(const std::vector<Material>& scene_materials)
        : materials(scene_materials), context(0), pipeline(0), stream(0),
          raygen_prog_group(nullptr), miss_prog_groups{nullptr, nullptr},
          hitgroup_prog_groups{nullptr, nullptr},
          d_raygen_record(0), d_miss_record(0),
          d_hitgroup_record(0), module(0), payload_buffer(nullptr) {}

OptixPipelineImpl::~OptixPipelineImpl() {
    cleanup();
}

void OptixPipelineImpl::cleanup() {
    if (pipeline) {
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = 0;
    }
    if (module) {
        OPTIX_CHECK(optixModuleDestroy(module));
        module = 0;
    }
    if (raygen_prog_group) {
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
        raygen_prog_group = nullptr;
    }
    if (miss_prog_groups[0]) {
        OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_groups[0]));
        miss_prog_groups[0] = nullptr;
    }
    if (miss_prog_groups[1]) {
        OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_groups[1]));
        miss_prog_groups[1] = nullptr;
    }
    if (hitgroup_prog_groups[0]) {
        OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_groups[0]));
        hitgroup_prog_groups[0] = nullptr;
    }
    if (hitgroup_prog_groups[1]) {
        OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_groups[1]));
        hitgroup_prog_groups[1] = nullptr;
    }
    if (d_raygen_record != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_raygen_record)));
        d_raygen_record = 0;
    }
    if (d_miss_record != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_miss_record)));
        d_miss_record = 0;
    }
    if (d_hitgroup_record != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hitgroup_record)));
        d_hitgroup_record = 0;
    }
    if (payload_buffer != nullptr) {
        CUDA_CHECK(cudaFree(payload_buffer));
        payload_buffer = nullptr;
    }
    if (context) {
        OPTIX_CHECK(optixDeviceContextDestroy(context));
        context = 0;
    }
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = 0;
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

PathTracerPipeline::PathTracerPipeline()
        : impl(nullptr) // Initialize the unique_ptr to nullptr
{
    // Constructor does not perform initialization
}

PathTracerPipeline::~PathTracerPipeline() {
    // Destructor automatically cleans up the unique_ptr
}

void PathTracerPipeline::initialize(const std::vector<Material>& materials) {
    if (impl) {
        throw std::runtime_error("PathTracerPipeline has already been initialized.");
    }
    impl = std::make_unique<OptixPipelineImpl>(materials);
    impl->initialize();
}

void PathTracerPipeline::render(const LaunchParams& params) {
    if (!impl) {
        throw std::runtime_error("PathTracerPipeline is not initialized.");
    }
    impl->render(params);
}

OptixDeviceContext PathTracerPipeline::getContext() const {
    if (!impl) {
        throw std::runtime_error("PathTracerPipeline is not initialized.");
    }
    return impl->getContext();
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
    pipeline_compile_options.numPayloadValues = 2; // Two 32-bit payload values (pointer)
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

    // Shadow hit group
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

    if (sizeof_log > 1) {
        std::cout << "Raygen Program Group Log: " << log << std::endl;
    }

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

    if (sizeof_log > 1) {
        std::cout << "Miss Program Group Log: " << log << std::endl;
    }

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

    if (sizeof_log > 1) {
        std::cout << "Hitgroup Program Group Log: " << log << std::endl;
    }

    std::cout << "Program groups created successfully." << std::endl;
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
    pipeline_link_options.maxTraceDepth = 10;  // Allow for multiple bounces
    // Removed 'debugLevel' as it is not a member of OptixPipelineLinkOptions

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

void OptixPipelineImpl::allocatePayloadBuffer(unsigned int width, unsigned int height) {
    size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t payload_size = sizeof(Payload) * num_pixels;

    // Allocate device memory for Payloads
    CUDA_CHECK(cudaMalloc(&payload_buffer, payload_size));

    // Initialize Payloads to zero
    CUDA_CHECK(cudaMemset(payload_buffer, 0, payload_size));

    std::cout << "Payload buffer allocated and initialized (" << num_pixels << " Payloads)" << std::endl;
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
    miss_records[0].data.bg_color = make_float3(0.0f, 0.0f, 0.2f); // Set background color (dark blue)

    // Shadow miss record
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[1], &miss_records[1]));
    // No additional data for shadow miss record

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissSbtRecord) * 2));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), miss_records, sizeof(MissSbtRecord) * 2, cudaMemcpyHostToDevice));

    // Create hitgroup records
    size_t num_materials = materials.size();
    std::vector<RadianceHitGroupSbtRecord> radiance_hitgroup_records(num_materials);
    std::vector<ShadowHitGroupSbtRecord> shadow_hitgroup_records(num_materials);

    for (size_t i = 0; i < num_materials; ++i) {
        // Radiance hit group record
        RadianceHitGroupSbtRecord& radiance_record = radiance_hitgroup_records[i];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[RAY_TYPE_RADIANCE], &radiance_record));
        radiance_record.material = materials[i];

        // Shadow hit group record
        ShadowHitGroupSbtRecord& shadow_record = shadow_hitgroup_records[i];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[RAY_TYPE_SHADOW], &shadow_record));
        // No additional data for shadow hitgroup
    }

    // Allocate and copy radiance hitgroup records to device memory
    CUdeviceptr d_radiance_hitgroup_records;
    size_t radiance_size = sizeof(RadianceHitGroupSbtRecord) * radiance_hitgroup_records.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radiance_hitgroup_records), radiance_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radiance_hitgroup_records), radiance_hitgroup_records.data(), radiance_size, cudaMemcpyHostToDevice));

    // Allocate and copy shadow hitgroup records to device memory
    CUdeviceptr d_shadow_hitgroup_records;
    size_t shadow_size = sizeof(ShadowHitGroupSbtRecord) * shadow_hitgroup_records.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shadow_hitgroup_records), shadow_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_shadow_hitgroup_records), shadow_hitgroup_records.data(), shadow_size, cudaMemcpyHostToDevice));

    // Combine radiance and shadow hitgroup records into a single buffer
    size_t total_hitgroup_records = radiance_hitgroup_records.size() + shadow_hitgroup_records.size();
    size_t combined_hitgroup_size = radiance_size + shadow_size;

    CUdeviceptr d_combined_hitgroup_records;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_combined_hitgroup_records), combined_hitgroup_size));

    // Copy all hitgroup records into the combined buffer
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_combined_hitgroup_records), radiance_hitgroup_records.data(), radiance_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_combined_hitgroup_records + radiance_size), shadow_hitgroup_records.data(), shadow_size, cudaMemcpyHostToDevice));

    // Initialize SBT
    sbt = {};

    sbt.raygenRecord = d_raygen_record;

    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 2;

    sbt.hitgroupRecordBase = d_combined_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = sizeof(RadianceHitGroupSbtRecord); // Assuming uniform stride
    sbt.hitgroupRecordCount = static_cast<unsigned int>(total_hitgroup_records);

    // Debug: Output HitGroup SBT Records
    std::cout << "HitGroup SBT Records (" << total_hitgroup_records << " records):" << std::endl;
    for (size_t i = 0; i < radiance_hitgroup_records.size(); ++i) {
        const RadianceHitGroupSbtRecord& record = radiance_hitgroup_records[i];
        std::cout << "SBT Record " << i << " (Radiance): Material Type = " << record.material.type << std::endl;
    }
    for (size_t i = 0; i < shadow_hitgroup_records.size(); ++i) {
        const ShadowHitGroupSbtRecord& record = shadow_hitgroup_records[i];
        std::cout << "SBT Record " << (radiance_hitgroup_records.size() + i) << " (Shadow): No Material" << std::endl;
    }

    std::cout << "Shader Binding Table created successfully" << std::endl;
}


OptixDeviceContext OptixPipelineImpl::getContext() const {
    return context;
}

void OptixPipelineImpl::render(const LaunchParams& params) {
    // Allocate Payload Buffer if not already allocated or if dimensions have changed
    static unsigned int last_width = 0;
    static unsigned int last_height = 0;
    if (params.frame.width != last_width || params.frame.height != last_height || payload_buffer == nullptr) {
        if (payload_buffer != nullptr) {
            CUDA_CHECK(cudaFree(payload_buffer));
            payload_buffer = nullptr;
        }
        allocatePayloadBuffer(params.frame.width, params.frame.height);
        last_width = params.frame.width;
        last_height = params.frame.height;
    }

    // Allocate and copy LaunchParams to device
    void* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(LaunchParams)));

    // Update LaunchParams with the device pointer to Payload buffer
    LaunchParams host_params = params;
    host_params.payload_buffer = payload_buffer;

    CUDA_CHECK(cudaMemcpy(d_params, &host_params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

    std::cout << "Launching OptiX pipeline..." << std::endl;

    // Launch the pipeline
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

    // Synchronize the stream to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free LaunchParams memory
    CUDA_CHECK(cudaFree(d_params));

    std::cout << "Render completed successfully" << std::endl;
}
