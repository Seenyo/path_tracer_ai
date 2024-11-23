#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <random>
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
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        if (aperture <= 0.0f) {
            // Pinhole camera model
            return Ray(
                position,
                glm::normalize(lowerLeftCorner + s*horizontal + t*vertical - position)
            );
        } else {
            // Thin lens model for depth of field
            glm::vec2 ray_direction;
            do {
                ray_direction = glm::vec2(dist(gen), dist(gen));
            } while (glm::dot(ray_direction, ray_direction) >= 1.0f);
            ray_direction *= lensRadius;

            glm::vec3 offset = u * ray_direction.x + v * ray_direction.y;
            return Ray(
                position + offset,
                glm::normalize(lowerLeftCorner + s*horizontal + t*vertical - (position + offset))
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
};
