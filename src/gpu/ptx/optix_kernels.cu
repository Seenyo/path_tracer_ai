#include <optix.h>
#include "gpu/optix_types.hpp"
#include "gpu/cuda_utils_device.hpp"

// Constants
#define PI 3.14159265358979323846f

// Declare launch parameters as a global constant
__constant__ LaunchParams launchParams;

// Helper functions
__device__ __forceinline__ float3 getWorldRayOrigin() {
    return optixGetWorldRayOrigin();
}

__device__ __forceinline__ float3 getWorldRayDirection() {
    return optixGetWorldRayDirection();
}

__device__ __forceinline__ float3 getWorldNormalVector() {
    return normalize(make_float3(
        __int_as_float(optixGetAttribute_0()),
        __int_as_float(optixGetAttribute_1()),
        __int_as_float(optixGetAttribute_2())
    ));
}

__device__ __forceinline__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ void* unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

__device__ __forceinline__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

__device__ __forceinline__ RayPayload* getPayload() {
    return reinterpret_cast<RayPayload*>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
}

// Ray generation program
extern "C" __global__ void __raygen__rg() {
    // Get launch index
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int pixelIndex = idx.y * dim.x + idx.x;

    // Initialize random seed
    unsigned int seed = launchParams.seeds[pixelIndex];

    // Initialize accumulation color
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // Generate multiple samples per pixel
    for (int sample = 0; sample < launchParams.samplesPerPixel; ++sample) {
        // Calculate ray direction through pixel
        float u = (idx.x + random(seed)) / (dim.x - 1);
        float v = (idx.y + random(seed)) / (dim.y - 1);

        // Get camera ray
        float3 origin = launchParams.camera.position;
        float3 direction;

        // Calculate ray direction using camera parameters
        float theta = launchParams.camera.fov * PI / 180.0f;
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = viewport_height * launchParams.camera.aspectRatio;

        float3 w = normalize(-launchParams.camera.forward);
        float3 u_vec = normalize(cross(launchParams.camera.up, w));
        float3 v_vec = cross(w, u_vec);

        float3 horizontal = viewport_width * u_vec;
        float3 vertical = viewport_height * v_vec;
        float3 lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - w;

        direction = normalize(lower_left_corner + u*horizontal + v*vertical - origin);

        // Initialize payload
        RayPayload payload;
        payload.radiance = make_float3(0.0f, 0.0f, 0.0f);
        payload.attenuation = make_float3(1.0f, 1.0f, 1.0f);
        payload.origin = origin;
        payload.direction = direction;
        payload.depth = 0;
        payload.seed = seed;
        payload.materialId = -1;

        // Pack payload pointers
        unsigned int p0, p1;
        packPointer(&payload, p0, p1);

        optixTrace(
            launchParams.traversable,
            origin,
            direction,
            0.001f,                // tmin
            1e16f,                 // tmax
            0.0f,                  // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0,                     // SBT offset
            1,                     // SBT stride
            0,                     // missSBTIndex
            p0, p1
        );

        // Accumulate color
        color = color + payload.radiance;
    }

    // Average samples and store final color
    color = color / static_cast<float>(launchParams.samplesPerPixel);
    launchParams.colorBuffer[pixelIndex] = make_float4(color.x, color.y, color.z, 1.0f);
    launchParams.seeds[pixelIndex] = seed;
}

// Miss program
extern "C" __global__ void __miss__ms() {
    RayPayload* payload = getPayload();
    payload->radiance = make_float3(0.0f, 0.0f, 0.0f);
}

// Closest hit program
extern "C" __global__ void __closesthit__ch() {
    RayPayload* payload = getPayload();

    // Get hit point information
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const int primitiveIndex = optixGetPrimitiveIndex();
    const float3 rayDirection = getWorldRayDirection();
    const float3 hitPoint = getWorldRayOrigin() + optixGetRayTmax() * rayDirection;

    // Get material ID and properties
    const int materialId = payload->materialId;
    const GPUMaterial& material = launchParams.materials[materialId];

    // Calculate surface normal
    const float3 normal = getWorldNormalVector();

    // Direct lighting calculation
    float3 directLight = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < launchParams.numLights; ++i) {
        const GPULight& light = launchParams.lights[i];
        
        // Calculate light direction and distance
        float3 lightDir = light.position - hitPoint;
        float lightDistance = length(lightDir);
        lightDir = normalize(lightDir);

        // Shadow ray
        bool inShadow = false;
        unsigned int shadowPayload = 0;
        optixTrace(
            launchParams.traversable,
            hitPoint,
            lightDir,
            0.001f,
            lightDistance - 0.001f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            1,  // SBT offset for shadow ray
            1,  // SBT stride
            1,  // missSBTIndex
            shadowPayload
        );

        if (!inShadow) {
            float cosTheta = fmaxf(dot(normal, lightDir), 0.0f);
            float attenuation = light.intensity / (lightDistance * lightDistance);
            directLight = directLight + light.color * material.albedo * cosTheta * attenuation;
        }
    }

    // Handle different material types
    float3 newDirection;
    float3 attenuation;

    switch (material.type) {
        case 0: // DIFFUSE
            newDirection = randomHemisphereDirection(normal, payload->seed);
            attenuation = material.albedo;
            break;

        case 1: // SPECULAR
            newDirection = reflect(rayDirection, normal);
            if (material.roughness > 0.0f) {
                newDirection = normalize(newDirection + material.roughness * randomInUnitSphere(payload->seed));
            }
            attenuation = material.albedo;
            break;

        case 2: // DIELECTRIC
            float cosTheta = fminf(dot(-rayDirection, normal), 1.0f);
            float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
            float etai = 1.0f, etat = material.ior;
            float3 outwardNormal = normal;
            
            if (cosTheta < 0.0f) {
                cosTheta = -cosTheta;
                float temp = etai;
                etai = etat;
                etat = temp;
                outwardNormal = -normal;
            }
            
            float ratio = etai / etat;
            float r0 = (etai - etat) / (etai + etat);
            r0 = r0 * r0;
            float schlick = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
            
            if (ratio * sinTheta > 1.0f || random(payload->seed) < schlick) {
                newDirection = reflect(rayDirection, outwardNormal);
            } else {
                newDirection = refract(rayDirection, outwardNormal, ratio);
            }
            attenuation = make_float3(1.0f, 1.0f, 1.0f);
            break;
    }

    // Update payload for next bounce
    if (payload->depth < launchParams.maxBounces) {
        payload->origin = hitPoint;
        payload->direction = newDirection;
        payload->attenuation = payload->attenuation * attenuation;
        payload->radiance = payload->radiance + directLight * payload->attenuation;
        payload->depth++;

        // Pack payload pointers
        unsigned int p0, p1;
        packPointer(payload, p0, p1);

        optixTrace(
            launchParams.traversable,
            hitPoint,
            newDirection,
            0.001f,
            1e16f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            p0, p1
        );
    }
}

// Any hit program (for shadow rays)
extern "C" __global__ void __anyhit__ah() {
    optixTerminateRay();
}
