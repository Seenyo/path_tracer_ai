#include <optix.h>
#include "../../include/optix/launch_params.hpp"
#include "../../include/math/vec_math.hpp"
#include <stdio.h>

extern "C" __constant__ LaunchParams launch_params;

// Helper functions to pack/unpack pointers - only used for radiance rays
static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& u0, uint32_t& u1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    u0 = static_cast<uint32_t>(uptr >> 32);
    u1 = static_cast<uint32_t>(uptr & 0xFFFFFFFF);
}

static __forceinline__ __device__ void* unpackPointer(uint32_t u0, uint32_t u1) {
    const uint64_t uptr = (static_cast<uint64_t>(u0) << 32) | static_cast<uint64_t>(u1);
    return reinterpret_cast<void*>(uptr);
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
    } while (dot(p,p) >= 1.0f);
    return p;
}

static __forceinline__ __device__ float3 random_unit_vector(uint32_t& state) {
    return normalize(random_in_unit_sphere(state));
}

static __forceinline__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx)/(1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f - r0)*powf((1.0f - cosine),5.0f);
}

// Raygen program
extern "C" __global__ void __raygen__pinhole() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    size_t pixel_idx = static_cast<size_t>(idx.y) * dim.x + idx.x;
    if (pixel_idx >= (static_cast<size_t>(dim.x)*dim.y)) {
        return;
    }

    uint32_t seed = launch_params.random.seed + (uint32_t)pixel_idx + launch_params.frame_number * 1000000;

    float2 pixel = make_float2(
            (float)idx.x / (float)dim.x,
            (float)idx.y / (float)dim.y
    );
    float2 ndc = make_float2(2.0f * pixel.x - 1.0f, 2.0f * pixel.y - 1.0f);

    float3 forward = launch_params.camera.direction;
    float3 right = normalize(cross(forward, launch_params.camera.up));
    float3 up = normalize(cross(right, forward));

    float tan_fov_x = tanf(launch_params.camera.fov.x * 0.5f);
    float tan_fov_y = tanf(launch_params.camera.fov.y * 0.5f);

    float3 ray_direction = normalize(
            forward +
            right * (ndc.x * tan_fov_x) +
            up * (ndc.y * tan_fov_y)
    );
    float3 ray_origin = launch_params.camera.position;

    Payload* payload = launch_params.payload_buffer + pixel_idx;
    payload->radiance = make_float3(0.0f);
    payload->throughput = make_float3(1.0f);
    payload->seed = seed;
    payload->hit_tmax = 0.0f;
    payload->next_direction = make_float3(0.0f);

    uint32_t u0, u1;
    packPointer(payload, u0, u1);

    optixTrace(
            launch_params.traversable,
            ray_origin,
            ray_direction,
            0.001f,
            1e16f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,
            2,
            RAY_TYPE_RADIANCE,
            u0, u1
    );

    float3 ray_radiance = payload->radiance;
    float3 ray_throughput = payload->throughput;
    float3 accumulated_radiance = ray_radiance * ray_throughput;

    if (launch_params.frame_number == 0) {
        launch_params.frame.color_buffer[pixel_idx] = accumulated_radiance;
    } else {
        float a = 1.0f/(float)(launch_params.frame_number + 1);
        float3 old_color = launch_params.frame.color_buffer[pixel_idx];
        launch_params.frame.color_buffer[pixel_idx] = old_color + (accumulated_radiance - old_color)*a;
    }
}

// Miss program for radiance rays
extern "C" __global__ void __miss__radiance() {
    uint32_t u0 = optixGetPayload_0();
    uint32_t u1 = optixGetPayload_1();
    Payload* payload = reinterpret_cast<Payload*>(unpackPointer(u0, u1));

    payload->radiance += launch_params.miss.bg_color * payload->throughput;
}

// Miss program for shadow rays
extern "C" __global__ void __miss__shadow() {
    // For shadow rays, no pointer. Payload_0 is occlusion (0 = no occlusion).
    // Do nothing here, occluded stays 0.
}

// Closest hit for radiance rays
extern "C" __global__ void __closesthit__radiance() {
    uint32_t u0 = optixGetPayload_0();
    uint32_t u1 = optixGetPayload_1();
    Payload* payload = reinterpret_cast<Payload*>(unpackPointer(u0, u1));

    const HitGroupSbtRecord* hitgroup_data = reinterpret_cast<HitGroupSbtRecord*>(optixGetSbtDataPointer());
    const Material& material = hitgroup_data->material;

    float2 barycentrics = optixGetTriangleBarycentrics();
    int prim_idx = optixGetPrimitiveIndex();

    float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_direction = optixGetWorldRayDirection();
    float tmax = optixGetRayTmax();
    float3 p = ray_origin + ray_direction*tmax;
    const float3* normals = launch_params.geometry.normals + 3 * prim_idx;
    float3 n = normalize(
            normals[0]*(1.0f - barycentrics.x - barycentrics.y) +
            normals[1]*barycentrics.x +
            normals[2]*barycentrics.y
    );

    float3 nl = faceforward(n, -ray_direction);

    float3 emitted_radiance = make_float3(0.0f);
    float3 direct_lighting = make_float3(0.0f);
    float3 next_direction = make_float3(0.0f);

    switch (material.type) {
        case EMISSIVE: {
            emitted_radiance = material.emission;
            break;
        }
        case LAMBERTIAN: {
            // Direct lighting
            unsigned int num_lights = launch_params.geometry.num_lights;
            for (unsigned int i = 0; i < num_lights; ++i) {
                const Light& light = launch_params.geometry.lights[i];
                float3 light_dir = light.position - p;
                float distance = length(light_dir);
                light_dir = normalize(light_dir);

                // Shadow ray: Just boolean occlusion check
                uint32_t occluded = 0u;
                uint32_t dummy = 0u;
                optixTrace(
                        launch_params.traversable,
                        p + nl*0.001f,
                        light_dir,
                        0.001f,
                        distance - 0.001f,
                        0.0f,
                        OptixVisibilityMask(1),
                        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                        RAY_TYPE_SHADOW,
                        2,
                        RAY_TYPE_SHADOW,
                        occluded,
                        dummy
                );

                if (occluded == 0u) {
                    float NdotL = fmaxf(dot(nl, light_dir), 0.0f);
                    float3 brdf = material.base_color / M_PI;
                    float3 radiance = light.color * light.intensity / (distance*distance);
                    direct_lighting += brdf * radiance * NdotL;
                }
            }

            // Sample diffuse reflection for indirect lighting
            next_direction = normalize(nl + random_unit_vector(payload->seed));
            payload->seed = pcg_random(payload->seed);
            break;
        }
        case METAL: {
            float3 reflected = reflect(ray_direction, nl);
            next_direction = normalize(reflected + material.roughness * random_in_unit_sphere(payload->seed));
            payload->seed = pcg_random(payload->seed);

            float NdotV = dot(nl, -ray_direction);
            float3 F = material.f0 + (make_float3(1.0f) - material.f0)*powf(1.0f - NdotV,5.0f);
            payload->throughput *= F;
            break;
        }
        case DIELECTRIC: {
            float3 unit_direction = normalize(ray_direction);
            float cos_theta = fminf(dot(-unit_direction, nl),1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
            float ref_idx = material.ior;
            float etai_over_etat = (dot(unit_direction, nl)>0.0f) ? ref_idx : (1.0f/ref_idx);

            bool cannot_refract = etai_over_etat*sin_theta>1.0f;
            float reflect_prob = schlick(cos_theta, etai_over_etat);

            if (cannot_refract || random_float(payload->seed)<reflect_prob) {
                next_direction = reflect(unit_direction, nl);
            } else {
                float3 refracted;
                refract(unit_direction, nl, etai_over_etat, refracted);
                next_direction = refracted;
            }
            payload->seed = pcg_random(payload->seed);
            break;
        }
        default:
            // No-op
            break;
    }

    payload->radiance += emitted_radiance;
    payload->radiance += direct_lighting * payload->throughput;
    if (material.type == LAMBERTIAN) {
        payload->throughput *= material.base_color;
    }

    payload->hit_tmax = tmax;
    payload->next_direction = next_direction;
}

// Closest hit for shadow rays
extern "C" __global__ void __closesthit__shadow() {
    // For shadow rays, we don't use a pointer payload.
    // Just set payload_0 = 1 to indicate occlusion.
    optixSetPayload_0(1);
}
