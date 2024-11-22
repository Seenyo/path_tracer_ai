#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "ray.hpp"

class Camera {
public:
    Camera(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up, float fov)
        : position(position)
        , forward(glm::normalize(target - position))
        , up(glm::normalize(up))
        , fov(fov) {
        right = glm::normalize(glm::cross(forward, this->up));
        this->up = glm::cross(right, forward);
    }

    Ray getRay(float u, float v) const {
        float theta = glm::radians(fov);
        float h = glm::tan(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = viewport_height * (16.0f / 9.0f);  // Assuming 16:9 aspect ratio

        glm::vec3 horizontal = viewport_width * right;
        glm::vec3 vertical = viewport_height * up;
        glm::vec3 lower_left_corner = position - horizontal/2.0f - vertical/2.0f + forward;

        return Ray(position, glm::normalize(lower_left_corner + u*horizontal + v*vertical - position));
    }

    // Accessor methods
    const glm::vec3& getPosition() const { return position; }
    const glm::vec3& getForward() const { return forward; }
    const glm::vec3& getRight() const { return right; }
    const glm::vec3& getUp() const { return up; }
    float getFOV() const { return fov; }

private:
    glm::vec3 position;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    float fov;
};
