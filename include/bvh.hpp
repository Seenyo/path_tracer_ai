#pragma once

#include <vector>
#include <algorithm>
#include <memory>
#include <glm/glm.hpp>
#include "ray.hpp"
#include "intersection.hpp"
#include "triangle.hpp"

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() : min(std::numeric_limits<float>::max()), max(-std::numeric_limits<float>::max()) {}
    AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    bool intersect(const Ray& ray, float& tMin, float& tMax) const {
        glm::vec3 invD = 1.0f / ray.direction;
        glm::vec3 t0 = (min - ray.origin) * invD;
        glm::vec3 t1 = (max - ray.origin) * invD;
        
        glm::vec3 tSmaller = glm::min(t0, t1);
        glm::vec3 tBigger = glm::max(t0, t1);
        
        tMin = glm::max(tMin, glm::max(tSmaller.x, glm::max(tSmaller.y, tSmaller.z)));
        tMax = glm::min(tMax, glm::min(tBigger.x, glm::min(tBigger.y, tBigger.z)));
        
        return tMax > tMin;
    }

    AABB merge(const AABB& other) const {
        return AABB(
            glm::min(min, other.min),
            glm::max(max, other.max)
        );
    }
};

// Simple chunk-based BVH node
struct BVHChunk {
    AABB bounds;
    std::vector<Triangle> triangles;
};

class BVH {
public:
    BVH() = default;

    void build(std::vector<Triangle>& triangles) {
        if (triangles.empty()) {
            std::cout << "Warning: No triangles to build BVH" << std::endl;
            return;
        }

        std::cout << "Building simplified BVH with " << triangles.size() << " triangles..." << std::endl;

        // Compute overall bounds
        AABB totalBounds;
        for (const auto& tri : triangles) {
            totalBounds = totalBounds.merge(AABB(
                glm::min(glm::min(tri.v0, tri.v1), tri.v2),
                glm::max(glm::max(tri.v0, tri.v1), tri.v2)
            ));
        }
        std::cout << "Computed overall bounds" << std::endl;

        // Divide triangles into chunks
        const size_t CHUNK_SIZE = 1024;  // Larger chunks for simplicity
        size_t numChunks = (triangles.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        chunks.reserve(numChunks);

        for (size_t i = 0; i < triangles.size(); i += CHUNK_SIZE) {
            BVHChunk chunk;
            size_t end = std::min(i + CHUNK_SIZE, triangles.size());
            
            // Copy triangles for this chunk
            chunk.triangles.insert(chunk.triangles.end(),
                                 triangles.begin() + i,
                                 triangles.begin() + end);
            
            // Compute chunk bounds
            for (const auto& tri : chunk.triangles) {
                chunk.bounds = chunk.bounds.merge(AABB(
                    glm::min(glm::min(tri.v0, tri.v1), tri.v2),
                    glm::max(glm::max(tri.v0, tri.v1), tri.v2)
                ));
            }
            
            chunks.push_back(std::move(chunk));
            
            if (chunks.size() % 10 == 0) {
                std::cout << "Created " << chunks.size() << " chunks..." << std::endl;
            }
        }

        std::cout << "BVH construction completed with " << chunks.size() << " chunks" << std::endl;
    }

    bool intersect(const Ray& ray, Intersection& isect) const {
        bool hit = false;
        float closest = ray.tMax;

        // Check each chunk
        for (const auto& chunk : chunks) {
            float tMin = ray.tMin;
            float tMax = closest;

            // Skip if ray doesn't hit chunk bounds
            if (!chunk.bounds.intersect(ray, tMin, tMax)) {
                continue;
            }

            // Check triangles in chunk
            for (const auto& tri : chunk.triangles) {
                Intersection tempIsect;
                if (tri.intersect(ray, tempIsect) && tempIsect.t < closest) {
                    isect = tempIsect;
                    closest = tempIsect.t;
                    hit = true;
                }
            }
        }

        return hit;
    }

private:
    std::vector<BVHChunk> chunks;
};
