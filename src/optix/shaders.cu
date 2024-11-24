#include <optix.h>
#include "../include/optix/launch_params.hpp"
#include "../include/math/vec_math.hpp"

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

static __forceinline__ __device__ float3 random_in_hemisphere(uint32_t& state, const float3& normal) {
    float3 in_unit_sphere = random_in_unit_sphere(state);
    return dot(in_unit_sphere, normal) > 0.0f ? in_unit_sphere : -in_unit_sphere;
}

static __forceinline__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// Ray generation program
extern "C" __global__ void __raygen__pinhole() {
    // Get launch index
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Initialize random state
    uint32_t seed = launch_params.random.seed + idx.y * dim.x + idx.x;
    
    // Calculate ray direction
    const float2 pixel = make_float2(
        (float)idx.x / (float)dim.x,
        (float)idx.y / (float)dim.y
    );
    
    const float2 ndc = make_float2(2.0f * pixel.x - 1.0f, 2.0f * pixel.y - 1.0f);
    const float aspect = (float)dim.x / (float)dim.y;
    
    // Calculate ray direction using camera parameters
    const float3 forward = launch_params.camera.direction;
    const float3 right = normalize(cross(forward, launch_params.camera.up));
    const float3 up = normalize(cross(right, forward));
    
    const float tan_fov_x = tanf(launch_params.camera.fov.x * 0.5f);
    const float tan_fov_y = tanf(launch_params.camera.fov.y * 0.5f);
    
    const float3 direction = normalize(
        forward +
        right * (ndc.x * tan_fov_x * aspect) +
        up * (ndc.y * tan_fov_y)
    );
    
    // Initialize path state
    float3 ray_origin = launch_params.camera.position;
    float3 ray_direction = direction;
    float3 throughput = make_float3(1.0f);
    float3 radiance = make_float3(0.0f);
    uint32_t depth = 0;
    
    // Trace path
    while (depth < launch_params.frame.max_bounces) {
        // Trace ray
        unsigned int p0, p1;
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
            2,                     // SBT stride
            0,                     // missSBTIndex
            p0, p1                 // payload variables
        );
        
        // Get hit information
        float3 hit_radiance = make_float3(
            __uint_as_float(p0),
            __uint_as_float(p1),
            0.0f
        );
        
        // Accumulate radiance
        radiance += hit_radiance * throughput;
        
        // Update ray state for next bounce
        ray_origin = make_float3(
            __uint_as_float(p0),
            __uint_as_float(p1),
            0.0f
        );
        
        // Russian roulette termination
        if (depth > 3) {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if (random_float(seed) > p) break;
            throughput *= 1.0f / p;
        }
        
        depth++;
    }
    
    // Write result
    const unsigned int pixel_idx = idx.y * dim.x + idx.x;
    if (launch_params.frame_number == 0) {
        // First frame, just store the result
        launch_params.frame.color_buffer[pixel_idx] = radiance;
    } else {
        // Accumulate with previous frames
        const float a = 1.0f / (float)(launch_params.frame_number + 1);
        const float3 old_color = launch_params.frame.color_buffer[pixel_idx];
        launch_params.frame.color_buffer[pixel_idx] = lerp(old_color, radiance, a);
    }
}

// Miss program for radiance rays
extern "C" __global__ void __miss__radiance() {
    // Return background color
    float3 bg_color = launch_params.miss.bg_color;
    optixSetPayload_0(__float_as_uint(bg_color.x));
    optixSetPayload_1(__float_as_uint(bg_color.y));
}

// Miss program for shadow rays
extern "C" __global__ void __miss__shadow() {
    // Ray reached light source, no occlusion
    optixSetPayload_0(__float_as_uint(1.0f));  // Shadow not occluded
}

// Closest hit program for radiance rays
extern "C" __global__ void __closesthit__radiance() {
    // Get hit point information
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const int prim_idx = optixGetPrimitiveIndex();
    
    // Get material
    const Material& material = launch_params.geometry.materials[prim_idx];
    
    // Get hit point position and interpolate normal
    const float3 p = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    const float3* normals = launch_params.geometry.normals + 3 * prim_idx;
    const float3 n = normalize(
        normals[0] * (1.0f - barycentrics.x - barycentrics.y) +
        normals[1] * barycentrics.x +
        normals[2] * barycentrics.y
    );
    
    // Get random state from launch params
    uint32_t seed = launch_params.random.seed;
    
    // Initialize radiance
    float3 radiance = make_float3(0.0f);
    float3 next_origin = p + n * 0.001f;  // Offset slightly to avoid self-intersection
    
    // Handle material interaction
    switch (material.type) {
        case EMISSIVE: {
            radiance = material.emission;
            break;
        }
        case LAMBERTIAN: {
            // Sample diffuse reflection
            const float3 next_direction = normalize(n + random_unit_vector(seed));
            
            // Add direct lighting contribution
            for (unsigned int i = 0; i < launch_params.geometry.num_lights; ++i) {
                const Light& light = launch_params.geometry.lights[i];
                
                // Sample point on light
                const float3 light_pos = light.position + light.radius * random_in_unit_sphere(seed);
                const float3 light_dir = normalize(light_pos - p);
                const float light_dist = length(light_pos - p);
                
                // Check visibility
                unsigned int shadow_occluded = 0;
                optixTrace(
                    launch_params.traversable,
                    p + n * 0.001f,  // Offset to avoid self-intersection
                    light_dir,
                    0.001f,                // tmin
                    light_dist - 0.001f,   // tmax
                    0.0f,                  // rayTime
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                    1,                     // SBT offset (shadow ray)
                    2,                     // SBT stride
                    1,                     // missSBTIndex
                    shadow_occluded
                );
                
                if (!shadow_occluded) {
                    const float cos_theta = max(0.0f, dot(n, light_dir));
                    const float3 light_color = light.color * light.intensity;
                    radiance += material.base_color * light_color * cos_theta / (light_dist * light_dist);
                }
            }
            break;
        }
        case METAL: {
            // Perfect specular reflection with roughness
            const float3 reflected = reflect(optixGetWorldRayDirection(), n);
            const float3 next_direction = normalize(reflected + material.roughness * random_in_unit_sphere(seed));
            break;
        }
        case DIELECTRIC: {
            // Glass-like material with reflection and refraction
            const float3 unit_direction = normalize(optixGetWorldRayDirection());
            const float cos_theta = min(dot(-unit_direction, n), 1.0f);
            const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            const float etai_over_etat = dot(unit_direction, n) > 0 ? material.ior : 1.0f / material.ior;
            
            float3 next_direction;
            if (etai_over_etat * sin_theta > 1.0f || random_float(seed) < schlick(cos_theta, etai_over_etat)) {
                // Must reflect
                next_direction = reflect(unit_direction, n);
            } else {
                // Can refract
                next_direction = normalize(make_float3(
                    etai_over_etat * unit_direction.x + (etai_over_etat * cos_theta - sqrtf(1.0f - etai_over_etat * etai_over_etat * (1.0f - cos_theta * cos_theta))) * n.x,
                    etai_over_etat * unit_direction.y + (etai_over_etat * cos_theta - sqrtf(1.0f - etai_over_etat * etai_over_etat * (1.0f - cos_theta * cos_theta))) * n.y,
                    etai_over_etat * unit_direction.z + (etai_over_etat * cos_theta - sqrtf(1.0f - etai_over_etat * etai_over_etat * (1.0f - cos_theta * cos_theta))) * n.z
                ));
            }
            break;
        }
    }
    
    // Return radiance contribution and next ray origin
    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
}

// Closest hit program for shadow rays
extern "C" __global__ void __closesthit__shadow() {
    // Ray hit something, light is occluded
    optixSetPayload_0(__float_as_uint(0.0f));  // Shadow occluded
}
