#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include "../math/vec_math.hpp"
#include "launch_params.hpp"

class OptixScene {
public:
    OptixScene();
    ~OptixScene();

    void addTriangle(const float3& v0, const float3& v1, const float3& v2);
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

    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<Material> materials;
    std::vector<Light> lights;

    CUdeviceptr d_vertices;
    CUdeviceptr d_normals;
    CUdeviceptr d_materials;
    CUdeviceptr d_lights;
    OptixTraversableHandle gas_handle;

    CUdeviceptr d_sbt_index_buffer{0};
    CUdeviceptr d_temp_buffer{0};
    CUdeviceptr d_gas_output_buffer{0};
};