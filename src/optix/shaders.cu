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
    float3 throughput;
    uint32_t seed;
    float hit_tmax;
    float3 next_direction;  // Add this member
};

// Ray generation program
extern "C" __global__ void __raygen__pinhole() {
    // Get launch index
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Calculate pixel index
    const unsigned int pixel_idx = idx.y * dim.x + idx.x;

    // Initialize random state as uint32_t
    uint32_t seed = launch_params.random.seed + pixel_idx + launch_params.frame_number * 1000000;

    // Calculate ray direction
    const float2 pixel = make_float2(
            (float)idx.x / (float)dim.x,
            (float)idx.y / (float)dim.y
    );

    const float2 ndc = make_float2(2.0f * pixel.x - 1.0f, 2.0f * pixel.y - 1.0f);

    // Calculate ray direction using camera parameters
    const float3 forward = launch_params.camera.direction;
    const float3 right = normalize(cross(forward, launch_params.camera.up));
    const float3 up = normalize(cross(right, forward));

    const float tan_fov_x = tanf(launch_params.camera.fov.x * 0.5f);
    const float tan_fov_y = tanf(launch_params.camera.fov.y * 0.5f);

    float3 ray_direction = normalize(
            forward +
            right * (ndc.x * tan_fov_x) +
            up * (ndc.y * tan_fov_y)
    );

    // Initialize path state
    float3 ray_origin = launch_params.camera.position;
    float3 accumulated_radiance = make_float3(0.0f);
    float3 throughput = make_float3(1.0f);
    uint32_t depth = 0;

    // Path tracing loop
    while (depth < launch_params.frame.max_bounces) {
        // Initialize payload
        Payload payload;
        payload.radiance = make_float3(0.0f);
        payload.throughput = throughput;
        payload.seed = seed;
        payload.hit_tmax = 0.0f;
        payload.next_direction = make_float3(0.0f);  // Initialize

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
                RAY_TYPE_RADIANCE,     // SBT offset
                RAY_TYPE_COUNT,        // SBT stride
                RAY_TYPE_RADIANCE,     // missSBTIndex
                u0, u1                 // Payloads
        );

        // Accumulate radiance
        accumulated_radiance += payload.radiance * throughput;

        // Update throughput for the next bounce
        throughput *= payload.throughput;

        // Update seed
        seed = payload.seed;

        // Update ray origin and direction
        ray_origin = ray_origin + ray_direction * payload.hit_tmax;
        ray_direction = payload.hit_tmax > 0.0f ? payload.next_direction : make_float3(0.0f);

        // Russian roulette termination
        float max_component = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (max_component < 0.01f || depth >= launch_params.frame.max_bounces) {
            break;
        }

        if (random_float(seed) > max_component) {
            break;
        }
        throughput /= max_component;

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
    // Light is visible if miss is reached
    optixSetPayload_0(0u);
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

    // Ensure normal is facing the correct direction
    const float3 nl = faceforward(n, -ray_direction);

    // Initialize variables for next bounce
    float3 next_origin = p + nl * 0.001f;  // Offset to avoid self-intersection
    float3 next_direction = make_float3(0.0f);
    float3 emitted_radiance = make_float3(0.0f);
    float3 direct_lighting = make_float3(0.0f);

    // Handle material interaction
    switch (material.type) {
        case EMISSIVE: {
            emitted_radiance = material.emission;
            break;
        }
        case LAMBERTIAN: {
            // Direct lighting computation
            const unsigned int num_lights = launch_params.geometry.num_lights;
            for (unsigned int i = 0; i < num_lights; ++i) {
                const Light& light = launch_params.geometry.lights[i];
                float3 light_dir = light.position - p;
                float distance = length(light_dir);
                light_dir = normalize(light_dir);

                // Shadow ray payload
                uint32_t occluded = 0u;

                // Trace shadow ray
                optixTrace(
                        launch_params.traversable,
                        p + nl * 0.001f,
                        light_dir,
                        0.001f,
                        distance - 0.001f,
                        0.0f,
                        OptixVisibilityMask(1),
                        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                        RAY_TYPE_SHADOW,  // SBT offset for shadow rays
                        RAY_TYPE_COUNT,   // SBT stride
                        RAY_TYPE_SHADOW,  // missSBTIndex for shadow rays
                        occluded
                );

                // If not occluded, accumulate direct lighting
                if (occluded == 0u) {
                    float NdotL = max(dot(nl, light_dir), 0.0f);
                    float3 brdf = material.base_color / M_PI;  // Lambertian BRDF
                    float3 radiance = light.color * light.intensity / (distance * distance);
                    direct_lighting += brdf * radiance * NdotL;
                }
            }

            // Sample diffuse reflection for indirect lighting
            next_direction = normalize(nl + random_unit_vector(payload->seed));

            // Update seed
            payload->seed = pcg_random(payload->seed);
            break;
        }
        case METAL: {
            // Perfect specular reflection with roughness
            float3 reflected = reflect(ray_direction, nl);
            next_direction = normalize(reflected + material.roughness * random_in_unit_sphere(payload->seed));

            // Update seed
            payload->seed = pcg_random(payload->seed);

            // Update throughput based on Fresnel reflectance (simplified)
            float NdotV = dot(nl, -ray_direction);
            float3 F = material.f0 + (make_float3(1.0f) - material.f0) * powf(1.0f - NdotV, 5.0f);
            payload->throughput *= F;

            break;
        }
        case DIELECTRIC: {
            // Glass-like material with reflection and refraction
            float3 unit_direction = normalize(ray_direction);
            float cos_theta = min(dot(-unit_direction, nl), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            float ref_idx = material.ior;
            float etai_over_etat = dot(unit_direction, nl) > 0.0f ? ref_idx : 1.0f / ref_idx;

            bool cannot_refract = etai_over_etat * sin_theta > 1.0f;
            float reflect_prob = schlick(cos_theta, etai_over_etat);

            if (cannot_refract || random_float(payload->seed) < reflect_prob) {
                // Reflect
                next_direction = reflect(unit_direction, nl);
            } else {
                // Refract
                float3 refracted;
                refract(unit_direction, nl, etai_over_etat, refracted);
                next_direction = refracted;
            }

            // Update seed
            payload->seed = pcg_random(payload->seed);
            break;
        }
    }

    // Update payload radiance
    payload->radiance += emitted_radiance;
    payload->radiance += direct_lighting * payload->throughput;

    // Update throughput based on material
    if (material.type == LAMBERTIAN) {
        payload->throughput *= material.base_color;
    }

    // Update hit distance and next direction for next bounce
    payload->hit_tmax = tmax;
    payload->next_direction = next_direction;
}

// Closest hit program for shadow rays
extern "C" __global__ void __closesthit__shadow() {
    // Intersection found, light is occluded
    optixSetPayload_0(1u);
}
