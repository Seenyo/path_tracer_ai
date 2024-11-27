#include <optix.h>
#include "../include/optix/launch_params.hpp"
#include "../include/math/vec_math.hpp"

// Enable printf in device code
#include <stdio.h>

extern "C" {
__constant__ LaunchParams launch_params;
}

// Random number generation using PCG
static __forceinline__ __device__ uint32_t pcg_random(uint32_t& state) {
    uint32_t prev = state;
    state = prev * 747796405u + 2891336453u;
    uint32_t word = ((prev >> ((prev >> 28) + 4u)) ^ prev) * 277803737u;
    return (word >> 22) ^ word;
}

static __forceinline__ __device__ float random_float(uint32_t& state) {
    return pcg_random(state) * (1.0f / 4294967296.0f);
}

static __forceinline__ __device__ float3 random_in_unit_sphere(uint32_t& state) {
    float3 p;
    do {
        p = 2.0f * make_float3(random_float(state), random_float(state), random_float(state)) - make_float3(1.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

static __forceinline__ __device__ float3 random_unit_vector(uint32_t& state) {
    return normalize(random_in_unit_sphere(state));
}

static __forceinline__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// Helper functions to pack and unpack pointers
static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& u0, uint32_t& u1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    u0 = static_cast<uint32_t>(uptr >> 32);
    u1 = static_cast<uint32_t>(uptr & 0xFFFFFFFF);
}

static __forceinline__ __device__ void* unpackPointer(uint32_t u0, uint32_t u1) {
    const uint64_t uptr = (static_cast<uint64_t>(u0) << 32) | static_cast<uint64_t>(u1);
    return reinterpret_cast<void*>(uptr);
}

// Payload structure
struct Payload {
    float3 radiance;
    float throughput;
    uint32_t seed;
    float hit_tmax;
};

// Ray generation program without illegal payload access
extern "C" __global__ void __raygen__pinhole() {
    // Get launch index
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Calculate pixel index
    const unsigned int pixel_idx = idx.y * dim.x + idx.x;

    // Initialize random state as uint32_t
    uint32_t seed = launch_params.random.seed + pixel_idx;

    // Calculate ray direction
    const float2 pixel = make_float2(
            (float)idx.x / (float)dim.x,
            (float)idx.y / (float)dim.y
    );

    const float2 ndc = make_float2(2.0f * pixel.x - 1.0f, 2.0f * pixel.y - 1.0f);
//    const float aspect = (float)dim.x / (float)dim.y;

    // Calculate ray direction using camera parameters
    const float3 forward = launch_params.camera.direction;
    const float3 right = normalize(cross(forward, launch_params.camera.up));
    const float3 up = normalize(cross(right, forward));

    const float tan_fov_x = tanf(launch_params.camera.fov.x * 0.5f);
    const float tan_fov_y = tanf(launch_params.camera.fov.y * 0.5f);

    const float3 initial_direction = normalize(
            forward +
            right * (ndc.x * tan_fov_x) +
            up * (ndc.y * tan_fov_y)
    );

    // Initialize path state
    float3 ray_origin = launch_params.camera.position;
    float3 ray_direction = initial_direction;
    float throughput = 1.0f;  // Initial throughput is 1.0
    float3 accumulated_radiance = make_float3(0.0f);
    uint32_t depth = 0;

    // Progressive path tracing loop
    while (depth < launch_params.frame.max_bounces && throughput > 0.01f) {
        // Initialize payload
        Payload payload;
        payload.radiance = make_float3(0.0f);
        payload.throughput = throughput;
        payload.seed = seed;
        payload.hit_tmax = 0.0f;

        // Pack the pointer to the payload into two unsigned ints
        uint32_t u0, u1;
        packPointer(&payload, u0, u1);

        // Trace the ray, passing the payload pointer
        optixTrace(
                launch_params.traversable,
                ray_origin,
                ray_direction,
                0.001f,                // tmin
                1e16f,                 // tmax
                0.0f,                  // rayTime
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_NONE,
                0,                     // SBT offset
                0,                     // SBT stride
                0,                     // missSBTIndex
                u0, u1                 // Payloads
        );

        // Accumulate radiance
        accumulated_radiance += payload.radiance * payload.throughput;

        // Update throughput for the next bounce
        throughput = payload.throughput;

        // Update seed
        seed = payload.seed;

        // Update ray origin based on hit distance
        ray_origin += ray_direction * payload.hit_tmax;

        depth++;
    }

    // Write final accumulated radiance to the color buffer
    if (launch_params.frame_number == 0) {
        // First frame, just store the result
        launch_params.frame.color_buffer[pixel_idx] = accumulated_radiance;
    } else {
        // Accumulate with previous frames
        const float a = 1.0f / (float)(launch_params.frame_number + 1);
        const float3 old_color = launch_params.frame.color_buffer[pixel_idx];
        launch_params.frame.color_buffer[pixel_idx] = old_color + (accumulated_radiance - old_color) * a;
    }

    // Debug: Print final radiance
    // printf("Pixel %u - Final Radiance: (%f, %f, %f)\n", pixel_idx, accumulated_radiance.x, accumulated_radiance.y, accumulated_radiance.z);
}

// Miss program for radiance rays
extern "C" __global__ void __miss__radiance() {
    // Retrieve payload
    uint32_t u0 = optixGetPayload_0();
    uint32_t u1 = optixGetPayload_1();
    Payload* payload = reinterpret_cast<Payload*>(unpackPointer(u0, u1));

    // Add background color to radiance
    payload->radiance += launch_params.miss.bg_color * payload->throughput;
}

// Miss program for shadow rays
extern "C" __global__ void __miss__shadow() {
    // Not used in this example
}

// Closest hit program for radiance rays
extern "C" __global__ void __closesthit__radiance() {
    // Retrieve payload
    uint32_t u0 = optixGetPayload_0();
    uint32_t u1 = optixGetPayload_1();
    Payload* payload = reinterpret_cast<Payload*>(unpackPointer(u0, u1));

    // Retrieve the SBT data
    const HitGroupSbtRecord* hitgroup_data = reinterpret_cast<HitGroupSbtRecord*>(optixGetSbtDataPointer());
    const Material& material = hitgroup_data->material;

    // Get hit point information
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const int prim_idx = optixGetPrimitiveIndex();

    // Get hit point position and interpolate normal
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float tmax = optixGetRayTmax();
    const float3 p = ray_origin + ray_direction * tmax;
    const float3* normals = launch_params.geometry.normals + 3 * prim_idx;
    const float3 n = normalize(
            normals[0] * (1.0f - barycentrics.x - barycentrics.y) +
            normals[1] * barycentrics.x +
            normals[2] * barycentrics.y
    );

    // Initialize variables for next bounce
    float3 next_origin = p + n * 0.001f;  // Offset to avoid self-intersection
    float3 next_direction = make_float3(0.0f);
    float3 emitted_radiance = make_float3(0.0f);

    // Handle material interaction
    switch (material.type) {
        case EMISSIVE: {
            emitted_radiance = material.emission;
            break;
        }
        case LAMBERTIAN: {
            // Sample diffuse reflection
            next_direction = normalize(n + random_unit_vector(payload->seed));

            // Update seed
            payload->seed = pcg_random(payload->seed);

            // Accumulate direct lighting (if any)
            break;
        }
        case METAL: {
            // Perfect specular reflection with roughness
            float3 reflected = reflect(ray_direction, n);
            next_direction = normalize(reflected + material.roughness * random_in_unit_sphere(payload->seed));

            // Update seed
            payload->seed = pcg_random(payload->seed);
            break;
        }
        case DIELECTRIC: {
            // Glass-like material with reflection and refraction
            float3 unit_direction = normalize(ray_direction);
            float cos_theta = min(dot(-unit_direction, n), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            float etai_over_etat = (dot(unit_direction, n) > 0) ? material.ior : 1.0f / material.ior;

            float3 refracted;
            float reflect_prob;
            bool cannot_refract = etai_over_etat * sin_theta > 1.0f;
            if (cannot_refract) {
                reflect_prob = 1.0f;
            } else {
                reflect_prob = schlick(cos_theta, etai_over_etat);
            }

            if (cannot_refract || random_float(payload->seed) < reflect_prob) {
                // Reflect
                next_direction = reflect(unit_direction, n);
            } else {
                // Refract
                bool refracted_success = refract(unit_direction, n, etai_over_etat, refracted);
                if (refracted_success) {
                    next_direction = refracted;
                } else {
                    // Fallback to reflection if refraction fails
                    next_direction = reflect(unit_direction, n);
                }
            }

            // Update seed
            payload->seed = pcg_random(payload->seed);
            break;
        }
    }

    // Update throughput based on material
    switch (material.type) {
        case LAMBERTIAN:
            payload->throughput *= material.base_color.x; // Assuming scalar throughput
            break;
        case METAL:
            payload->throughput *= material.base_color.x; // Assuming scalar throughput
            break;
        case DIELECTRIC:
            // Optionally adjust throughput based on reflectance
            break;
        case EMISSIVE:
            // Emissive does not alter throughput
            break;
    }

    // Accumulate emitted radiance directly into payload radiance
    payload->radiance += emitted_radiance * payload->throughput;

    // Update hit distance for next bounce
    payload->hit_tmax = tmax;

    // Check if we should spawn a new ray
    if (payload->throughput > 0.01f) {
        // Pack the payload pointer again
        uint32_t u0, u1;
        packPointer(payload, u0, u1);

        // Trace the new ray
        optixTrace(
                launch_params.traversable,
                next_origin,          // New ray origin
                next_direction,       // New ray direction
                0.001f,               // tmin
                1e16f,                // tmax
                0.0f,                 // rayTime
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_NONE,
                0,                    // SBT offset
                0,                    // SBT stride
                0,                    // missSBTIndex
                u0, u1                // Payloads
        );
    }
}
