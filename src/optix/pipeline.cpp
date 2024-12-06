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

#include "../../include/optix/pipeline.hpp"
#include "../../include/optix/launch_params.hpp"

// Error check/report helper for OptiX
#define OPTIX_CHECK(call)                                                  \
    do {                                                                   \
        OptixResult res = call;                                            \
        if (res != OPTIX_SUCCESS) {                                        \
            std::stringstream ss;                                          \
            ss << "OptiX call '" << #call << "' failed: "                  \
               << optixGetErrorString(res);                                \
            throw std::runtime_error(ss.str());                            \
        }                                                                  \
    } while (0)

// Error check/report helper for CUDA
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                       \
            std::stringstream ss;                                         \
            ss << "CUDA call '" << #call << "' failed: "                  \
               << cudaGetErrorString(error) << std::endl;                 \
            throw std::runtime_error(ss.str());                           \
        }                                                                 \
    } while (0)

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
        optixPipelineDestroy(pipeline);
        pipeline = 0;
    }
    if (module) {
        optixModuleDestroy(module);
        module = 0;
    }
    if (raygen_prog_group) {
        optixProgramGroupDestroy(raygen_prog_group);
        raygen_prog_group = nullptr;
    }
    if (miss_prog_groups[0]) {
        optixProgramGroupDestroy(miss_prog_groups[0]);
        miss_prog_groups[0] = nullptr;
    }
    if (miss_prog_groups[1]) {
        optixProgramGroupDestroy(miss_prog_groups[1]);
        miss_prog_groups[1] = nullptr;
    }
    if (hitgroup_prog_groups[0]) {
        optixProgramGroupDestroy(hitgroup_prog_groups[0]);
        hitgroup_prog_groups[0] = nullptr;
    }
    if (hitgroup_prog_groups[1]) {
        optixProgramGroupDestroy(hitgroup_prog_groups[1]);
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
        optixDeviceContextDestroy(context);
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

    CUDA_CHECK(cudaStreamCreate(&stream));
}

PathTracerPipeline::PathTracerPipeline()
        : impl(nullptr) {}

PathTracerPipeline::~PathTracerPipeline() {
    // unique_ptr cleans up automatically
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
    CUDA_CHECK(cudaFree(0));

    CUdevice device;
    CUresult cu_res = cuDeviceGet(&device, 0);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error getting CUDA device");
    }

    CUcontext cu_ctx = nullptr;
    cu_res = cuCtxCreate(&cu_ctx, 0, device);
    if (cu_res != CUDA_SUCCESS) {
        throw std::runtime_error("Error creating CUDA context");
    }

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
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
}

void OptixPipelineImpl::createModule() {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
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
}

void OptixPipelineImpl::createProgramGroups() {
    // Raygen
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

    // Miss
    OptixProgramGroupDesc miss_prog_group_descs[2] = {};
    miss_prog_group_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_descs[0].miss.module = module;
    miss_prog_group_descs[0].miss.entryFunctionName = "__miss__radiance";

    miss_prog_group_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_descs[1].miss.module = module;
    miss_prog_group_descs[1].miss.entryFunctionName = "__miss__shadow";

    // Hitgroup
    OptixProgramGroupDesc hitgroup_prog_group_descs[2] = {};
    // Radiance hit group
    hitgroup_prog_group_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_descs[0].hitgroup.moduleCH = module;
    hitgroup_prog_group_descs[0].hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    // Shadow hit group
    hitgroup_prog_group_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_descs[1].hitgroup.moduleCH = module;
    hitgroup_prog_group_descs[1].hitgroup.entryFunctionNameCH = "__closesthit__shadow";

    OptixProgramGroupOptions program_group_options = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Raygen
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group
    ));

    // Miss groups
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
            context,
            miss_prog_group_descs,
            2,
            &program_group_options,
            log,
            &sizeof_log,
            miss_prog_groups
    ));

    // Hitgroup groups
    sizeof_log = sizeof(log);
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

void OptixPipelineImpl::createPipeline() {
    OptixProgramGroup program_groups[] = {
            raygen_prog_group,
            miss_prog_groups[0],
            miss_prog_groups[1],
            hitgroup_prog_groups[0],
            hitgroup_prog_groups[1]
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 10;

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

    OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            2048,
            2048,
            2048,
            1
    ));
}

void OptixPipelineImpl::allocatePayloadBuffer(unsigned int width, unsigned int height) {
    size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t payload_size = sizeof(Payload) * num_pixels;

    CUDA_CHECK(cudaMalloc(&payload_buffer, payload_size));
    CUDA_CHECK(cudaMemset(payload_buffer, 0, payload_size));
}

void OptixPipelineImpl::createSBT() {
    // Raygen record
    RayGenSbtRecord raygen_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &raygen_record));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &raygen_record, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    // Miss records
    MissSbtRecord miss_records[2] = {};
    // Radiance miss
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[0], &miss_records[0]));
    miss_records[0].data.bg_color = make_float3(0.0f, 0.0f, 0.2f);

    // Shadow miss
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[1], &miss_records[1]));
    // No extra data

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissSbtRecord)*2));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), miss_records, sizeof(MissSbtRecord)*2, cudaMemcpyHostToDevice));

    // Hitgroup records: we create a record for each (material * RAY_TYPE_COUNT)
    size_t num_materials = materials.size();
    std::vector<HitGroupSbtRecord> hitgroup_records(num_materials * RAY_TYPE_COUNT);

    for (size_t i = 0; i < num_materials; ++i) {
        // Radiance hit record
        HitGroupSbtRecord& radiance_record = hitgroup_records[i * 2 + RAY_TYPE_RADIANCE];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[RAY_TYPE_RADIANCE], &radiance_record));
        radiance_record.material = materials[i];

        // Shadow hit record
        HitGroupSbtRecord& shadow_record = hitgroup_records[i * 2 + RAY_TYPE_SHADOW];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[RAY_TYPE_SHADOW], &shadow_record));
        // We can reuse the same material or leave it as is
        shadow_record.material = materials[i];
    }

    size_t hitgroup_size = hitgroup_records.size() * sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), hitgroup_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), hitgroup_records.data(), hitgroup_size, cudaMemcpyHostToDevice));

    sbt = {};
    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 2;
    sbt.hitgroupRecordBase = d_hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
}

OptixDeviceContext OptixPipelineImpl::getContext() const {
    return context;
}

void OptixPipelineImpl::render(const LaunchParams& params) {
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

    void* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(LaunchParams)));
    LaunchParams host_params = params;
    host_params.payload_buffer = payload_buffer;
    CUDA_CHECK(cudaMemcpy(d_params, &host_params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
            pipeline,
            stream,
            reinterpret_cast<CUdeviceptr>(d_params),
            sizeof(LaunchParams),
            &sbt,
            params.frame.width,
            params.frame.height,
            1
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_params));
}
