#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <vector_types.h>
#include <glm/glm.hpp>

// Convert between GLM and CUDA types
inline __host__ __device__ float3 glmToCuda(const glm::vec3 &v)
{
    return make_float3(v.x, v.y, v.z);
}

inline __host__ __device__ glm::vec3 cudaToGlm(const float3 &v)
{
    return glm::vec3(v.x, v.y, v.z);
}

// GPU-compatible AABB structure
struct GPUAABB
{
    float3 min;
    float3 max;

    __host__ __device__ GPUAABB()
    {
        min = make_float3(1e20f, 1e20f, 1e20f);
        max = make_float3(-1e20f, -1e20f, -1e20f);
    }

    __host__ __device__ GPUAABB(const float3 &min, const float3 &max) : min(min), max(max) {}

    __device__ bool intersect(const float3 &origin, const float3 &direction, float &tMin, float &tMax) const
    {
        float3 invD = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
        float3 t0 = make_float3(
            (min.x - origin.x) * invD.x,
            (min.y - origin.y) * invD.y,
            (min.z - origin.z) * invD.z);
        float3 t1 = make_float3(
            (max.x - origin.x) * invD.x,
            (max.y - origin.y) * invD.y,
            (max.z - origin.z) * invD.z);

        float tminx = fminf(t0.x, t1.x);
        float tminy = fminf(t0.y, t1.y);
        float tminz = fminf(t0.z, t1.z);
        float tmaxx = fmaxf(t0.x, t1.x);
        float tmaxy = fmaxf(t0.y, t1.y);
        float tmaxz = fmaxf(t0.z, t1.z);

        tMin = fmaxf(fmaxf(tminx, tminy), fmaxf(tminz, tMin));
        tMax = fminf(fminf(tmaxx, tmaxy), fminf(tmaxz, tMax));

        return tMax > tMin;
    }

    __host__ __device__ GPUAABB merge(const GPUAABB &other) const
    {
        return GPUAABB(
            make_float3(
                fminf(min.x, other.min.x),
                fminf(min.y, other.min.y),
                fminf(min.z, other.min.z)),
            make_float3(
                fmaxf(max.x, other.max.x),
                fmaxf(max.y, other.max.y),
                fmaxf(max.z, other.max.z)));
    }

    __host__ __device__ int maxExtentAxis() const
    {
        float3 extent = make_float3(
            max.x - min.x,
            max.y - min.y,
            max.z - min.z);
        if (extent.x > extent.y && extent.x > extent.z)
            return 0;
        else if (extent.y > extent.z)
            return 1;
        else
            return 2;
    }
};

// GPU-compatible material structure
struct GPUMaterial
{
    int type;
    float3 albedo;
    float roughness;
    float metallic;
    float ior;
    float padding[2];

    // Remove constructor or use default
    __host__ __device__ GPUMaterial() = default;
};

// GPU-compatible light structure
struct GPULight
{
    float3 position;
    float3 color;
    float intensity;
    float padding;

    __host__ __device__ GPULight() = default;
};

// GPU-compatible camera structure
struct GPUCamera
{
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float fov;
    float aspectRatio;
    float padding[2];

    __host__ __device__ GPUCamera() = default;
};

// Launch parameters structure
struct LaunchParams
{
    // Output buffer
    float4 *colorBuffer;
    unsigned int width;
    unsigned int height;
    unsigned int samplesPerPixel;
    unsigned int maxBounces;
    float gamma;

    // Scene data
    OptixTraversableHandle traversable;
    GPULight *lights;
    unsigned int numLights;
    GPUMaterial *materials;
    unsigned int numMaterials;

    // Camera
    GPUCamera camera;

    // Random seed
    unsigned int *seeds;
};

// Ray payload structure
struct RayPayload
{
    float3 radiance;    // Accumulated radiance
    float3 attenuation; // Path throughput
    float3 origin;      // Origin for next bounce
    float3 direction;   // Direction for next bounce
    int depth;          // Current bounce depth
    unsigned int seed;  // Random seed
    int materialId;     // Hit material ID

    __device__ RayPayload()
        : radiance(make_float3(0.0f, 0.0f, 0.0f)), attenuation(make_float3(1.0f, 1.0f, 1.0f)), origin(make_float3(0.0f, 0.0f, 0.0f)), direction(make_float3(0.0f, 0.0f, 1.0f)), depth(0), seed(0), materialId(-1) {}
};

// Vertex attributes structure
struct VertexAttributes
{
    float3 position;
    float3 normal;
    float2 texCoord;
    int materialId;

    __host__ __device__ VertexAttributes()
        : position(make_float3(0.0f, 0.0f, 0.0f)), normal(make_float3(0.0f, 1.0f, 0.0f)), texCoord(make_float2(0.0f, 0.0f)), materialId(0) {}
};
