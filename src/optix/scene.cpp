#include "../../include/optix/scene.hpp"
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

// Error check/report helper for CUDA
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                      \
            std::string error_str = cudaGetErrorString(error);           \
            throw std::runtime_error(std::string("CUDA call '") +        \
                #call + "' failed: " + error_str);                       \
        }                                                                \
    } while (0)

// Error check/report helper for OptiX
#define OPTIX_CHECK(call)                                                \
    do {                                                                 \
        OptixResult res = call;                                         \
        if (res != OPTIX_SUCCESS) {                                     \
            std::string error_str = optixGetErrorName(res);             \
            throw std::runtime_error(std::string("OptiX call '") +      \
                #call + "' failed: " + error_str);                      \
        }                                                               \
    } while (0)

OptixScene::OptixScene() 
    : d_vertices(0), d_normals(0), d_materials(0), d_lights(0), d_gas_output_buffer(0), gas_handle(0) {}

OptixScene::~OptixScene() {
    cleanup();
}

void OptixScene::addTriangle(const float3& v0, const float3& v1, const float3& v2) {
    // Add vertices
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);

    // Calculate and add normal
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 normal = normalize(cross(edge1, edge2));
    normals.push_back(normal);
    normals.push_back(normal);
    normals.push_back(normal);
}

void OptixScene::setMaterials(const std::vector<Material>& new_materials) {
    materials = new_materials;
}

void OptixScene::setLights(const std::vector<Light>& new_lights) {
    lights = new_lights;
}

void OptixScene::uploadGeometry() {
    if (vertices.empty()) {
        throw std::runtime_error("No geometry to upload");
    }

    // Allocate and upload vertices
    const size_t vertices_size = vertices.size() * sizeof(float3);
    void* d_vertices_ptr;
    CUDA_CHECK(cudaMalloc(&d_vertices_ptr, vertices_size));
    d_vertices = reinterpret_cast<CUdeviceptr>(d_vertices_ptr);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    // Allocate and upload normals
    const size_t normals_size = normals.size() * sizeof(float3);
    void* d_normals_ptr;
    CUDA_CHECK(cudaMalloc(&d_normals_ptr, normals_size));
    d_normals = reinterpret_cast<CUdeviceptr>(d_normals_ptr);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_normals),
        normals.data(),
        normals_size,
        cudaMemcpyHostToDevice
    ));

    // Allocate and upload materials
    if (!materials.empty()) {
        const size_t materials_size = materials.size() * sizeof(Material);
        void* d_materials_ptr;
        CUDA_CHECK(cudaMalloc(&d_materials_ptr, materials_size));
        d_materials = reinterpret_cast<CUdeviceptr>(d_materials_ptr);
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_materials),
            materials.data(),
            materials_size,
            cudaMemcpyHostToDevice
        ));
    }

    // Allocate and upload lights
    if (!lights.empty()) {
        const size_t lights_size = lights.size() * sizeof(Light);
        void* d_lights_ptr;
        CUDA_CHECK(cudaMalloc(&d_lights_ptr, lights_size));
        d_lights = reinterpret_cast<CUdeviceptr>(d_lights_ptr);
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_lights),
            lights.data(),
            lights_size,
            cudaMemcpyHostToDevice
        ));
    }
}

void OptixScene::buildAcceleration(OptixDeviceContext context) {
    uploadGeometry();

    // Build input for acceleration structure
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Triangle build input
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    build_input.triangleArray.vertexBuffers = &d_vertices;

    // Create index buffer
    uint32_t num_triangles = static_cast<uint32_t>(vertices.size() / 3);
    std::vector<uint32_t> indices;
    indices.reserve(num_triangles * 3);
    for (uint32_t i = 0; i < num_triangles * 3; ++i) {
        indices.push_back(i);
    }

    // Upload index buffer to device
    CUdeviceptr d_indices;
    size_t index_buffer_size = indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), index_buffer_size));
    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_indices),
            indices.data(),
            index_buffer_size,
            cudaMemcpyHostToDevice
    ));

    // Set index buffer in build input
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    build_input.triangleArray.numIndexTriplets = num_triangles;
    build_input.triangleArray.indexBuffer = d_indices;

    // Prepare SBT index buffer
    std::vector<uint32_t> sbt_indices(num_triangles);
    for (uint32_t i = 0; i < num_triangles; ++i) {
        sbt_indices[i] = i; // Each triangle uses its own SBT record
    }

    // Upload SBT index buffer to device
    CUdeviceptr d_sbt_index_buffer;
    size_t sbt_index_buffer_size = sbt_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index_buffer), sbt_index_buffer_size));
    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_sbt_index_buffer),
            sbt_indices.data(),
            sbt_index_buffer_size,
            cudaMemcpyHostToDevice
    ));

    // Create triangle flags array
    std::vector<uint32_t> triangle_flags(num_triangles, OPTIX_GEOMETRY_FLAG_NONE);

    // Build input settings
    build_input.triangleArray.flags = triangle_flags.data(); // Host pointer
    build_input.triangleArray.numSbtRecords = num_triangles;
    build_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_index_buffer;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    // Build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Query build sizes
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            &build_input,
            1,  // num build inputs
            &gas_buffer_sizes
    ));

    // Allocate buffers
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    // Build acceleration structure
    OPTIX_CHECK(optixAccelBuild(
            context,
            0,                  // CUDA stream
            &accel_options,
            &build_input,
            1,                  // num build inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &gas_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
    ));

    // Clean up temp buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index_buffer)));
}



OptixTraversableHandle OptixScene::getTraversableHandle() const {
    return gas_handle;
}

CUdeviceptr OptixScene::getNormalsBuffer() const {
    return d_normals;
}

CUdeviceptr OptixScene::getMaterialsBuffer() const {
    return d_materials;
}

CUdeviceptr OptixScene::getLightsBuffer() const {
    return d_lights;
}

void OptixScene::cleanup() {
    if (d_vertices)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    if (d_normals)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_normals)));
    if (d_materials)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_materials)));
    if (d_lights)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lights)));
    if (d_gas_output_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
    if (d_sbt_index_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index_buffer)));

    d_vertices = 0;
    d_normals = 0;
    d_materials = 0;
    d_lights = 0;
    d_gas_output_buffer = 0;
    d_sbt_index_buffer = 0;
    gas_handle = 0;
}

