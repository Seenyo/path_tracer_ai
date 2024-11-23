#pragma once

#include <glm/glm.hpp>

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    float tMin = 0.001f;
    mutable float tMax = std::numeric_limits<float>::infinity();

    Ray(const glm::vec3& o, const glm::vec3& d) 
        : origin(o), direction(glm::normalize(d)) {}

    glm::vec3 at(float t) const {
        return origin + direction * t;
    }
};
