#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "../math/vec_math.hpp"

// Ensure proper alignment for the launch parameters
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
    unsigned int samples_per_pixel;  // Added for accumulation
    unsigned int max_bounces;        // Added for path tracing depth
};

// Camera parameters
struct ALIGN(16) Camera {
    float3 position;
    float pad1;              // Ensure 16-byte alignment
    float3 direction;
    float pad2;              // Ensure 16-byte alignment
    float3 up;
    float pad3;              // Ensure 16-byte alignment
    float2 fov;
    float2 pad4;             // Ensure 16-byte alignment
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
    float3 base_color;       // Diffuse color or specular color
    float metallic;          // 0 = dielectric, 1 = metal
    float3 emission;         // Emission color
    float roughness;         // Surface roughness
    float3 f0;              // Fresnel reflection at normal incidence
    float ior;              // Index of refraction
    MaterialType type;      // Material type
    float3 pad;            // Ensure 16-byte alignment
};

// Light source
struct ALIGN(16) Light {
    float3 position;         // Position of the light
    float intensity;         // Light intensity
    float3 color;           // Light color
    float radius;           // Light radius for soft shadows
};

// Miss shader parameters
struct ALIGN(16) MissData {
    float3 bg_color;
    float pad;               // Ensure 16-byte alignment
};

// Geometry data
struct ALIGN(16) GeometryData {
    float3* normals;         // Vertex normals buffer
    Material* materials;     // Material buffer
    Light* lights;          // Light sources
    unsigned int num_lights; // Number of lights
};

// Random state
struct ALIGN(16) RandomState {
    unsigned int seed;
    unsigned int pad[3];     // Ensure 16-byte alignment
};

// Launch parameters
struct ALIGN(16) LaunchParams {
    FrameBuffer frame;
    Camera camera;
    MissData miss;
    GeometryData geometry;
    RandomState random;
    OptixTraversableHandle traversable;
    unsigned int frame_number;  // Added for temporal accumulation
    unsigned int pad[2];       // Ensure overall 16-byte alignment
};

// SBT record structures
struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) RayGenSbtRecord {
    ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) MissSbtRecord {
    ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    MissData data;
};

struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupSbtRecord {
    ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    Material material;
};


// Ray types
enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW = 1,
    RAY_TYPE_COUNT
};