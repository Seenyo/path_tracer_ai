#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "ray.hpp"

class Camera {
public:
    Camera(
        const glm::vec3& position = glm::vec3(0.0f, 0.0f, 0.0f),
        const glm::vec3& lookAt = glm::vec3(0.0f, 0.0f, -1.0f),
        const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f),
        float verticalFOV = 45.0f,
        float aspectRatio = 16.0f/9.0f,
        float aperture = 0.0f,
        float focusDistance = 10.0f
    ) : position(position), aperture(aperture), focusDistance(focusDistance) {
        float theta = glm::radians(verticalFOV);
        float h = std::tan(theta/2.0f);
        float viewportHeight = 2.0f * h;
        float viewportWidth = aspectRatio * viewportHeight;

        w = glm::normalize(position - lookAt);
        u = glm::normalize(glm::cross(up, w));
        v = glm::cross(w, u);

        horizontal = focusDistance * viewportWidth * u;
        vertical = focusDistance * viewportHeight * v;
        lowerLeftCorner = position - horizontal/2.0f - vertical/2.0f - focusDistance * w;

        lensRadius = aperture / 2.0f;
    }

    Ray getRay(float s, float t) const {
        if (aperture <= 0.0f) {
            // Pinhole camera model
            return Ray(
                position,
                lowerLeftCorner + s*horizontal + t*vertical - position
            );
        } else {
            // Thin lens model for depth of field
            glm::vec2 rd = lensRadius * randomInUnitDisk();
            glm::vec3 offset = u * rd.x + v * rd.y;
            return Ray(
                position + offset,
                lowerLeftCorner + s*horizontal + t*vertical - (position + offset)
            );
        }
    }

private:
    glm::vec3 position;
    glm::vec3 lowerLeftCorner;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 u, v, w;
    float lensRadius;
    float aperture;
    float focusDistance;

    // Helper function to generate random points in a unit disk for DOF
    static glm::vec2 randomInUnitDisk() {
        while (true) {
            glm::vec2 p = 2.0f * glm::vec2(
                static_cast<float>(rand()) / RAND_MAX,
                static_cast<float>(rand()) / RAND_MAX
            ) - glm::vec2(1.0f);
            
            if (glm::dot(p, p) < 1.0f)
                return p;
        }
    }
};