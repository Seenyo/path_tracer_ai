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
        for (int i = 0; i < 3; i++) {
            float invD = 1.0f / ray.direction[i];
            float t0 = (min[i] - ray.origin[i]) * invD;
            float t1 = (max[i] - ray.origin[i]) * invD;
            if (invD < 0.0f) std::swap(t0, t1);
            
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            
            if (tMax <= tMin) return false;
        }
        return true;
    }

    AABB merge(const AABB& other) const {
        return AABB(
            glm::min(min, other.min),
            glm::max(max, other.max)
        );
    }

    float surfaceArea() const {
        glm::vec3 extent = max - min;
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
};

class BVHNode {
public:
    AABB bounds;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    std::vector<Triangle> triangles;
    bool isLeaf;

    BVHNode() : isLeaf(false) {}

    bool intersect(const Ray& ray, Intersection& isect) const {
        float tMin = ray.tMin;
        float tMax = ray.tMax;

        if (!bounds.intersect(ray, tMin, tMax)) {
            return false;
        }

        if (isLeaf) {
            bool hit = false;
            for (const auto& tri : triangles) {
                if (tri.intersect(ray, isect)) {
                    hit = true;
                    ray.tMax = isect.t;
                }
            }
            return hit;
        }

        bool hitLeft = left && left->intersect(ray, isect);
        bool hitRight = right && right->intersect(ray, isect);
        
        return hitLeft || hitRight;
    }
};

class BVH {
public:
    BVH() = default;

    void build(std::vector<Triangle>& triangles) {
        if (triangles.empty()) return;

        // Build the BVH tree
        root = buildRecursive(triangles, 0);
        
        std::cout << "BVH construction completed" << std::endl;
    }

    bool intersect(const Ray& ray, Intersection& isect) const {
        if (!root) return false;
        return root->intersect(ray, isect);
    }

private:
    std::unique_ptr<BVHNode> root;
    static constexpr int MAX_TRIANGLES_PER_LEAF = 4;
    static constexpr int MAX_DEPTH = 32;

    std::unique_ptr<BVHNode> buildRecursive(std::vector<Triangle>& triangles, int depth) {
        auto node = std::make_unique<BVHNode>();
        
        // Calculate bounds
        node->bounds = AABB();
        for (const auto& tri : triangles) {
            node->bounds = node->bounds.merge(AABB(
                glm::min(glm::min(tri.v0, tri.v1), tri.v2),
                glm::max(glm::max(tri.v0, tri.v1), tri.v2)
            ));
        }

        // Create leaf node if criteria met
        if (triangles.size() <= MAX_TRIANGLES_PER_LEAF || depth >= MAX_DEPTH) {
            node->isLeaf = true;
            node->triangles = triangles;
            return node;
        }

        // Find best split using SAH
        float bestCost = std::numeric_limits<float>::max();
        size_t bestSplit = 0;
        int bestAxis = 0;

        for (int axis = 0; axis < 3; ++axis) {
            // Sort triangles by centroid along current axis
            std::sort(triangles.begin(), triangles.end(), [axis](const Triangle& a, const Triangle& b) {
                return a.getCenter()[axis] < b.getCenter()[axis];
            });

            // Try different split positions
            float parentArea = node->bounds.surfaceArea();
            AABB leftBox;
            std::vector<float> leftAreas(triangles.size());
            
            // Precompute left areas
            for (size_t i = 0; i < triangles.size() - 1; ++i) {
                const auto& tri = triangles[i];
                leftBox = leftBox.merge(AABB(
                    glm::min(glm::min(tri.v0, tri.v1), tri.v2),
                    glm::max(glm::max(tri.v0, tri.v1), tri.v2)
                ));
                leftAreas[i] = leftBox.surfaceArea();
            }

            // Try each split position
            AABB rightBox;
            for (size_t i = triangles.size() - 1; i > 0; --i) {
                const auto& tri = triangles[i];
                rightBox = rightBox.merge(AABB(
                    glm::min(glm::min(tri.v0, tri.v1), tri.v2),
                    glm::max(glm::max(tri.v0, tri.v1), tri.v2)
                ));

                float leftArea = leftAreas[i - 1];
                float rightArea = rightBox.surfaceArea();
                
                float cost = 0.125f + (i * leftArea + (triangles.size() - i) * rightArea) / parentArea;

                if (cost < bestCost) {
                    bestCost = cost;
                    bestSplit = i;
                    bestAxis = axis;
                }
            }
        }

        // Sort along best axis if not already sorted
        if (bestAxis != 2) {
            std::sort(triangles.begin(), triangles.end(), [bestAxis](const Triangle& a, const Triangle& b) {
                return a.getCenter()[bestAxis] < b.getCenter()[bestAxis];
            });
        }

        // Split triangles
        std::vector<Triangle> leftTris(triangles.begin(), triangles.begin() + bestSplit);
        std::vector<Triangle> rightTris(triangles.begin() + bestSplit, triangles.end());

        // Create children
        node->left = buildRecursive(leftTris, depth + 1);
        node->right = buildRecursive(rightTris, depth + 1);

        return node;
    }
};
