#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include "ray.hpp"
#include "intersection.hpp"
#include "material.hpp"
#include "triangle.hpp"
#include "bvh.hpp"

// Forward declare tinyobj types we need
namespace tinyobj {
    struct attrib_t;
    struct shape_t;
    struct material_t;
    class ObjReader;
}

struct Light {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;

    Light(const glm::vec3& pos, const glm::vec3& col, float intens)
        : position(pos), color(col), intensity(intens) {
        // Validate light parameters
        if (intensity <= 0.0f) {
            std::cout << "Warning: Invalid light intensity " << intensity << ", setting to 1.0" << std::endl;
            intensity = 1.0f;
        }
    }
};

class Scene {
private:
    std::vector<Triangle> triangles;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<Light> lights;
    BVH bvh;

public:
    Scene() {
        std::cout << "Setting up lights..." << std::endl;
        
        // Add lights in a balanced configuration for the reoriented model
        
        // Main key light (front right, slightly higher)
        lights.emplace_back(
            glm::vec3(2.0f, 3.5f, 2.0f),     // Front right
            glm::vec3(1.0f, 0.95f, 0.8f),    // Warm white
            9.0f                             // Higher intensity for key light
        );
        
        // Fill light (front left, lower intensity)
        lights.emplace_back(
            glm::vec3(-1.5f, 2.0f, 1.5f),    // Front left
            glm::vec3(0.8f, 0.9f, 1.0f),     // Cool white
            2.0f                             // Lower intensity for fill
        );
        
        // Rim light (back)
        lights.emplace_back(
            glm::vec3(0.0f, 2.0f, -2.0f),    // Back
            glm::vec3(1.0f),                  // Pure white
            1.0f                             // Medium intensity for rim
        );

        // Ground bounce light (for softer shadows)
        lights.emplace_back(
            glm::vec3(0.0f, 0.1f, 0.0f),     // Near ground
            glm::vec3(0.9f, 0.9f, 1.0f),     // Slightly cool
            2.0f                              // Low intensity for subtle fill
        );

        for (size_t i = 0; i < lights.size(); ++i) {
            Light& light = lights[i];
            std::cout << "Light " << i << ":" << std::endl;
            std::cout << "  Position: (" << light.position.x << ", " 
                     << light.position.y << ", " << light.position.z << ")" << std::endl;
            std::cout << "  Color: (" << light.color.x << ", " 
                     << light.color.y << ", " << light.color.z << ")" << std::endl;
            std::cout << "  Intensity: " << light.intensity << std::endl;
        }
    }

    bool loadFromObj(const std::string& objPath);
    
    bool intersect(const Ray& ray, Intersection& isect) const {
        return bvh.intersect(ray, isect);
    }
    
    const std::vector<std::shared_ptr<Material>>& getMaterials() const { 
        return materials; 
    }
    
    const std::vector<Light>& getLights() const { 
        return lights; 
    }
};
