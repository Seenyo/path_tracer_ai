// launch_params.hpp

#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "../math/vec_math.hpp"

// Alignment macro
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define ALIGN(x) __align__(x)
#else
#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif
#endif

// Frame buffer
struct ALIGN(16) FrameBuffer {
    float3* color_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_pixel;
    unsigned int max_bounces;
};

// Camera parameters
struct ALIGN(16) Camera {
    float3 position;
    float pad1;
    float3 direction;
    float pad2;
    float3 up;
    float pad3;
    float2 fov;
    float2 pad4;
};

// Material types
enum MaterialType {
    LAMBERTIAN = 0,
    METAL = 1,
    DIELECTRIC = 2,
    EMISSIVE = 3
};

// Material properties
struct ALIGN(16) Material {
    float3 base_color;
    float metallic;
    float3 emission;
    float roughness;
    float3 f0;
    float ior;
    MaterialType type;
    float pad; // Ensure 16-byte alignment
};

// Light source
struct ALIGN(16) Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
};

// Miss shader parameters
struct ALIGN(16) MissData {
    float3 bg_color;
    float pad;
};

// Geometry data
struct ALIGN(16) GeometryData {
    float3* normals;
    Light* lights;
    unsigned int num_lights;
    unsigned int pad;
};

// Random state
struct ALIGN(16) RandomState {
    unsigned int seed;
    unsigned int pad[3];
};

// Payload structure
struct ALIGN(16) Payload {
    float3 radiance;        // 12 bytes
    float3 throughput;      // 12 bytes
    uint32_t seed;          // 4 bytes
    float hit_tmax;         // 4 bytes
    float3 next_direction;  // 12 bytes
    float pad;              // 4 bytes to make total size 48 bytes
};

// Launch parameters
struct ALIGN(16) LaunchParams {
    FrameBuffer frame;
    Camera camera;
    MissData miss;
    GeometryData geometry;
    RandomState random;
    OptixTraversableHandle traversable;
    unsigned int frame_number;
    unsigned int pad1;
    unsigned int pad2;
    Payload* payload_buffer; // Device pointer to Payloads
};

// SBT record structures for radiance and shadow
struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) RadianceHitGroupSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    Material material;
};

struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) ShadowHitGroupSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // No additional data needed for shadow hitgroups
};

// Ray Generation SBT Record
struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) RayGenSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // No additional data needed for raygen
};

// Miss SBT Record
struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) MissSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    MissData data; // Include MissData for radiance miss shader
};

// Ray types
enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW = 1,
    RAY_TYPE_COUNT
};
