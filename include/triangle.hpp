#pragma once

#include <glm/glm.hpp>
#include "ray.hpp"
#include "intersection.hpp"

struct Triangle {
    glm::vec3 v0, v1, v2;        // Vertices
    glm::vec3 n0, n1, n2;        // Vertex normals
    glm::vec2 uv0, uv1, uv2;     // Texture coordinates
    int materialId;

    Triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
            const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
            const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2,
            int matId)
        : v0(v0), v1(v1), v2(v2)
        , n0(n0), n1(n1), n2(n2)
        , uv0(uv0), uv1(uv1), uv2(uv2)
        , materialId(matId) {}

    bool intersect(const Ray& ray, Intersection& isect) const {
        const float EPSILON = 0.0000001f;
        
        // Compute edges
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        
        // Calculate determinant
        glm::vec3 h = glm::cross(ray.direction, edge2);
        float a = glm::dot(edge1, h);
        
        // Check if ray is parallel to triangle
        if (a > -EPSILON && a < EPSILON)
            return false;
        
        float f = 1.0f / a;
        glm::vec3 s = ray.origin - v0;
        float u = f * glm::dot(s, h);
        
        // Check if intersection lies outside triangle
        if (u < 0.0f || u > 1.0f)
            return false;
        
        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(ray.direction, q);
        
        // Check if intersection lies outside triangle
        if (v < 0.0f || u + v > 1.0f)
            return false;
        
        float t = f * glm::dot(edge2, q);
        
        // Check if intersection is behind ray origin or beyond max distance
        if (t < ray.tMin || t > ray.tMax)
            return false;
        
        // Compute intersection point and interpolate attributes
        float w = 1.0f - u - v;
        glm::vec3 normal = glm::normalize(w * n0 + u * n1 + v * n2);
        glm::vec2 uv = w * uv0 + u * uv1 + v * uv2;
        
        isect.set(t, ray.at(t), normal, uv, materialId);
        return true;
    }

    glm::vec3 getCenter() const {
        return (v0 + v1 + v2) / 3.0f;
    }
};
