#pragma once

#include <glm/glm.hpp>

struct Intersection {
    float t = std::numeric_limits<float>::infinity();  // Distance along ray
    glm::vec3 position;                               // Hit position
    glm::vec3 normal;                                 // Surface normal
    glm::vec2 uv;                                    // Texture coordinates
    int materialId = -1;                             // Material index
    bool hit = false;                                // Whether there was a hit

    Intersection() = default;

    void set(float _t, const glm::vec3& pos, const glm::vec3& norm, const glm::vec2& _uv, int matId) {
        t = _t;
        position = pos;
        normal = glm::normalize(norm);
        uv = _uv;
        materialId = matId;
        hit = true;
    }
};
