#pragma once

#include <vector>
#include <algorithm>
#include <memory>
#include <glm/glm.hpp>
#include "ray.hpp"
#include "intersection.hpp"
#include "triangle.hpp"
#include "aabb.hpp"

struct BVHNode {
    AABB bounds;
    BVHNode* left = nullptr;
    BVHNode* right = nullptr;
    std::vector<Triangle> triangles; // Only for leaf nodes

    bool isLeaf() const {
        return left == nullptr && right == nullptr;
    }
};

class BVH {
public:
    BVHNode* root = nullptr;

    void build(std::vector<Triangle>& triangles) {
        if (triangles.empty()) {
            std::cout << "Warning: No triangles to build BVH" << std::endl;
            return;
        }
        std::cout << "Building hierarchical BVH with " << triangles.size() << " triangles..." << std::endl;
        root = buildRecursive(triangles, 0, triangles.size());
        std::cout << "BVH construction completed." << std::endl;
    }

    bool intersect(const Ray& ray, Intersection& isect) const {
        return intersectNode(root, ray, isect);
    }

private:
    static constexpr int MAX_TRIANGLES_PER_LEAF = 8;

    BVHNode* buildRecursive(std::vector<Triangle>& tris, size_t start, size_t end) {
        BVHNode* node = new BVHNode();

        // Compute bounds for all triangles in [start, end)
        AABB bounds;
        for (size_t i = start; i < end; ++i) {
            bounds = bounds.merge(tris[i].getAABB());
        }
        node->bounds = bounds;

        size_t count = end - start;
        if (count <= MAX_TRIANGLES_PER_LEAF) {
            // Create leaf node
            node->triangles.insert(node->triangles.end(), tris.begin() + start, tris.begin() + end);
        } else {
            // Split along the largest axis
            int axis = bounds.maxExtentAxis();
            size_t mid = start + count / 2;

            std::nth_element(tris.begin() + start, tris.begin() + mid, tris.begin() + end,
                [axis](const Triangle& a, const Triangle& b) {
                    return a.getCenter()[axis] < b.getCenter()[axis];
                });

            node->left = buildRecursive(tris, start, mid);
            node->right = buildRecursive(tris, mid, end);
        }
        return node;
    }

    bool intersectNode(const BVHNode* node, const Ray& ray, Intersection& isect) const {
        if (!node) return false;

        float tMin = ray.tMin;
        float tMax = ray.tMax;
        if (!node->bounds.intersect(ray, tMin, tMax)) {
            return false;
        }

        bool hit = false;
        if (node->isLeaf()) {
            for (const auto& tri : node->triangles) {
                Intersection tempIsect;
                if (tri.intersect(ray, tempIsect)) {
                    if (tempIsect.t < isect.t) {
                        isect = tempIsect;
                        ray.tMax = tempIsect.t;
                        hit = true;
                    }
                }
            }
        } else {
            Intersection leftIsect, rightIsect;
            bool hitLeft = intersectNode(node->left, ray, leftIsect);
            bool hitRight = intersectNode(node->right, ray, rightIsect);

            if (hitLeft && hitRight) {
                if (leftIsect.t < rightIsect.t) {
                    isect = leftIsect;
                } else {
                    isect = rightIsect;
                }
                hit = true;
            } else if (hitLeft) {
                isect = leftIsect;
                hit = true;
            } else if (hitRight) {
                isect = rightIsect;
                hit = true;
            }
        }
        return hit;
    }
};

