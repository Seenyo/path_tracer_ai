#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

enum class MaterialType {
    DIFFUSE,
    SPECULAR,
    DIELECTRIC
};

struct Material {
    MaterialType type = MaterialType::DIFFUSE;
    glm::vec3 albedo = glm::vec3(0.8f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float ior = 1.5f;  // Index of refraction (for dielectric materials)
};

namespace MaterialUtils {
    inline float schlickFresnel(float cosTheta, float F0) {
        float x = 1.0f - cosTheta;
        float x2 = x * x;
        float x5 = x2 * x2 * x;
        return F0 + (1.0f - F0) * x5;
    }

    inline float ggxDistribution(float NdotH, float roughness) {
        if (roughness < 0.0f) roughness = 0.0f;
        if (roughness > 1.0f) roughness = 1.0f;

        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        
        float NdotH2 = NdotH * NdotH;
        float denom = NdotH2 * (alpha2 - 1.0f) + 1.0f;
        
        if (denom <= 0.0f) return 0.0f;
        
        float D = alpha2 / (glm::pi<float>() * denom * denom);
        return D;
    }

    inline float geometrySchlickGGX(float NdotV, float roughness) {
        float r = roughness + 1.0f;
        float k = (r * r) / 8.0f;
        float denom = NdotV * (1.0f - k) + k;
        
        if (denom <= 0.0f) return 0.0f;
        return NdotV / denom;
    }

    inline float geometrySmith(float NdotV, float NdotL, float roughness) {
        float ggx2 = geometrySchlickGGX(NdotV, roughness);
        float ggx1 = geometrySchlickGGX(NdotL, roughness);
        return ggx1 * ggx2;
    }
};
