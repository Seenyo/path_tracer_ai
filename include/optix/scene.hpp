#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include "../math/vec_math.hpp"
#include "launch_params.hpp"

// Ensure proper alignment for OptiX types
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define ALIGN(x) __align__(x)
#else
#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif
#endif

class OptixScene {
public:
    OptixScene();
    ~OptixScene();

    // Updated addTriangle function with material index parameter
    void addTriangle(const float3& v0, const float3& v1, const float3& v2, uint32_t material_idx);

    void setMaterials(const std::vector<Material>& materials);
    void setLights(const std::vector<Light>& lights);
    void buildAcceleration(OptixDeviceContext context);

    OptixTraversableHandle getTraversableHandle() const;
    CUdeviceptr getNormalsBuffer() const;
    CUdeviceptr getMaterialsBuffer() const;
    CUdeviceptr getLightsBuffer() const;

private:
    void uploadGeometry();
    void cleanup();

    // Geometry data
    std::vector<float3> vertices;
    std::vector<float3> normals;

    // Added material indices per triangle
    std::vector<uint32_t> material_indices;

    // Materials and lights
    std::vector<Material> materials;
    std::vector<Light> lights;

    // Device pointers
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_normals = 0;
    CUdeviceptr d_materials = 0;
    CUdeviceptr d_lights = 0;

    // Acceleration structure handle
    OptixTraversableHandle gas_handle = 0;

    // Device memory for acceleration structure
    CUdeviceptr d_gas_output_buffer = 0;
};
