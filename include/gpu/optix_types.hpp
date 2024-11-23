#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#ifndef __CUDA_ARCH__
#include <glm/glm.hpp>

// Convert between GLM and CUDA types
inline HOST_DEVICE float3 glmToCuda(const glm::vec3 &v)
{
    return make_float3(v.x, v.y, v.z);
}

inline HOST_DEVICE glm::vec3 cudaToGlm(const float3 &v)
{
    return glm::vec3(v.x, v.y, v.z);
}

#else
// Device-side versions without GLM
inline __device__ float3 glmToCuda(const float3 &v)
{
    return v;
}

#endif // __CUDA_ARCH__

// GPU-compatible material structure
struct GPUMaterial
{
    int type;
    float3 albedo;
    float roughness;
    float metallic;
    float ior;
    float padding[2];

    HOST_DEVICE GPUMaterial() = default;
};

// GPU-compatible light structure
struct GPULight
{
    float3 position;
    float3 color;
    float intensity;
    float padding;

    HOST_DEVICE GPULight() = default;
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

    HOST_DEVICE GPUCamera() = default;
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

    HOST_DEVICE RayPayload()
        : radiance(make_float3(0.0f, 0.0f, 0.0f)),
          attenuation(make_float3(1.0f, 1.0f, 1.0f)),
          origin(make_float3(0.0f, 0.0f, 0.0f)),
          direction(make_float3(0.0f, 0.0f, 1.0f)),
          depth(0),
          seed(0),
          materialId(-1) {}
};

// Vertex attributes structure
struct VertexAttributes
{
    float3 position;
    float3 normal;
    float2 texCoord;
    int materialId;

    HOST_DEVICE VertexAttributes()
        : position(make_float3(0.0f, 0.0f, 0.0f)),
          normal(make_float3(0.0f, 1.0f, 0.0f)),
          texCoord(make_float2(0.0f, 0.0f)),
          materialId(0) {}
};
