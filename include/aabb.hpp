#pragma once

#include <glm/glm.hpp>
#include "ray.hpp"

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() : min(std::numeric_limits<float>::max()), max(-std::numeric_limits<float>::max()) {}
    AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    bool intersect(const Ray& ray, float& tMin, float& tMax) const {
        for (int a = 0; a < 3; ++a) {
            float invD = 1.0f / ray.direction[a];
            float t0 = (min[a] - ray.origin[a]) * invD;
            float t1 = (max[a] - ray.origin[a]) * invD;
            if (invD < 0.0f) std::swap(t0, t1);
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            if (tMax <= tMin)
                return false;
        }
        return true;
    }

    AABB merge(const AABB& other) const {
        return AABB(
            glm::min(min, other.min),
            glm::max(max, other.max)
        );
    }

    int maxExtentAxis() const {
        glm::vec3 extent = max - min;
        if (extent.x > extent.y && extent.x > extent.z) return 0;
        else if (extent.y > extent.z) return 1;
        else return 2;
    }
};
