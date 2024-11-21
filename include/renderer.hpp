#pragma once

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <atomic>
#include <omp.h>
#include <cmath>
#include <glm/glm.hpp>
#include "scene.hpp"
#include "camera.hpp"

class Renderer {
public:
    struct Settings {
        int width;
        int height;
        int samplesPerPixel;
        int maxBounces;
        float gamma;

        Settings() 
            : width(800)
            , height(450)
            , samplesPerPixel(10)
            , maxBounces(3)
            , gamma(2.2f) {}
    };

    Renderer(const Settings& settings = Settings())
        : settings(settings)
        , rng(std::random_device{}())
        , distribution(0.0f, 1.0f) {
        frameBuffer.resize(settings.width * settings.height, glm::vec3(0.0f));
        std::cout << "Renderer initialized with " << settings.width << "x" << settings.height 
                  << " resolution" << std::endl;
    }

    void render(const Scene& scene, const Camera& camera) {
        std::cout << "\nStarting render with settings:" << std::endl;
        std::cout << "Resolution: " << settings.width << "x" << settings.height << std::endl;
        std::cout << "Samples per pixel: " << settings.samplesPerPixel << std::endl;
        std::cout << "Max bounces: " << settings.maxBounces << std::endl;

        std::atomic<int> pixelsCompleted{0};
        std::atomic<int> lastPercentage{0};
        int totalPixels = settings.width * settings.height;

        std::cout << "Starting pixel rendering..." << std::endl;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int y = 0; y < settings.height; ++y) {
            // Create a thread-local random number generator
            std::mt19937 localRng(std::random_device{}() + y);
            std::uniform_real_distribution<float> localDist(0.0f, 1.0f);

            for (int x = 0; x < settings.width; ++x) {
                glm::vec3 color(0.0f);
                bool hasValidSample = false;
                
                for (int s = 0; s < settings.samplesPerPixel; ++s) {
                    float u = (x + localDist(localRng)) / (settings.width - 1);
                    float v = (y + localDist(localRng)) / (settings.height - 1);
                    
                    Ray ray = camera.getRay(u, v);
                    glm::vec3 sample = tracePath(ray, scene, 0);
                    
                    if (isValidColor(sample, "Sample computation")) {
                        color += sample;
                        hasValidSample = true;
                    }
                }
                
                if (hasValidSample) {
                    color /= static_cast<float>(settings.samplesPerPixel);
                } else {
                    color = glm::vec3(1.0f, 0.0f, 1.0f); // Debug color for invalid pixels
                }
                
                frameBuffer[y * settings.width + x] = color;

                // Update progress
                int completed = ++pixelsCompleted;
                int percentage = (completed * 100) / totalPixels;
                
                if (percentage > lastPercentage) {
                    int oldPercentage = lastPercentage.load();
                    if (percentage > oldPercentage && 
                        lastPercentage.compare_exchange_strong(oldPercentage, percentage)) {
                        #pragma omp critical
                        {
                            std::cout << "\rRendering progress: " << percentage << "% (" 
                                     << completed << "/" << totalPixels << " pixels)" << std::flush;
                        }
                    }
                }
            }
        }
        
        std::cout << "\nRendering completed" << std::endl;
    }

    void saveImage(const std::string& filename);

private:
    Settings settings;
    std::vector<glm::vec3> frameBuffer;
    std::mt19937 rng;
    std::uniform_real_distribution<float> distribution;

    bool isValidColor(const glm::vec3& color, const char* location) {
        if (std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z) ||
            std::isinf(color.x) || std::isinf(color.y) || std::isinf(color.z)) {
            #pragma omp critical
            {
                std::cout << "Invalid color at " << location << ": "
                         << color.x << ", " << color.y << ", " << color.z << std::endl;
            }
            return false;
        }
        return true;
    }

    float randomFloat() {
        return distribution(rng);
    }

    glm::vec3 tracePath(const Ray& ray, const Scene& scene, int depth) {
        if (depth >= settings.maxBounces) {
            return glm::vec3(0.0f);
        }

        Intersection isect;
        if (!scene.intersect(ray, isect)) {
            return glm::vec3(0.0f); // Background color (black for now)
        }

        isect.normal = glm::normalize(isect.normal);

        const auto& materials = scene.getMaterials();
        if (isect.materialId >= materials.size()) {
            #pragma omp critical
            {
                std::cout << "Warning: Invalid material ID " << isect.materialId << std::endl;
            }
            return glm::vec3(1.0f, 0.0f, 1.0f); // Debug color for invalid materials
        }

        const auto& material = materials[isect.materialId];
        if (!material) {
            #pragma omp critical
            {
                std::cout << "Warning: Null material pointer" << std::endl;
            }
            return glm::vec3(1.0f, 0.0f, 1.0f); // Debug color for null materials
        }

        // Direct lighting calculation
        glm::vec3 directLight = calculateDirectLighting(isect, scene, -ray.direction);
        if (!isValidColor(directLight, "Direct lighting")) {
            return glm::vec3(0.0f);
        }

        // Handle different material types
        switch (material->type) {
            case MaterialType::DIFFUSE: {
                glm::vec3 randomDir = randomHemisphereDirection(isect.normal);
                Ray bounceRay(isect.position + isect.normal * 0.001f, randomDir);
                
                float cosTheta = glm::dot(randomDir, isect.normal);
                if (std::isnan(cosTheta) || std::isinf(cosTheta)) {
                    #pragma omp critical
                    {
                        std::cout << "Invalid cosTheta in diffuse reflection" << std::endl;
                    }
                    return glm::vec3(0.0f);
                }

                glm::vec3 brdf = material->albedo / glm::pi<float>();
                glm::vec3 indirect = tracePath(bounceRay, scene, depth + 1);
                
                if (!isValidColor(indirect, "Indirect diffuse")) {
                    return glm::vec3(0.0f);
                }

                return directLight + brdf * indirect * cosTheta * 2.0f * glm::pi<float>();
            }

            case MaterialType::SPECULAR: {
                glm::vec3 reflected = glm::reflect(ray.direction, isect.normal);
                if (material->roughness > 0.0f) {
                    reflected = glm::normalize(reflected + material->roughness * randomInUnitSphere());
                }
                Ray bounceRay(isect.position + isect.normal * 0.001f, reflected);
                
                float cosTheta = glm::dot(reflected, isect.normal);
                if (std::isnan(cosTheta) || std::isinf(cosTheta)) {
                    #pragma omp critical
                    {
                        std::cout << "Invalid cosTheta in specular reflection" << std::endl;
                    }
                    return glm::vec3(0.0f);
                }

                glm::vec3 indirect = tracePath(bounceRay, scene, depth + 1);
                if (!isValidColor(indirect, "Indirect specular")) {
                    return glm::vec3(0.0f);
                }

                return directLight + material->albedo * indirect * cosTheta;
            }

            case MaterialType::DIELECTRIC: {
                float cosTheta = glm::dot(-ray.direction, isect.normal);
                float etai = 1.0f, etat = material->ior;
                glm::vec3 normal = isect.normal;
                
                if (cosTheta < 0.0f) {
                    cosTheta = -cosTheta;
                    std::swap(etai, etat);
                    normal = -normal;
                }
                
                float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
                float ratio = etai / etat;
                
                glm::vec3 direction;
                if (ratio * sinTheta > 1.0f || 
                    randomFloat() < MaterialUtils::schlickFresnel(cosTheta, (etai - etat) / (etai + etat))) {
                    direction = glm::reflect(ray.direction, normal);
                } else {
                    direction = glm::refract(ray.direction, normal, ratio);
                }
                
                if (std::isnan(glm::length(direction)) || std::isinf(glm::length(direction))) {
                    #pragma omp critical
                    {
                        std::cout << "Invalid direction in dielectric" << std::endl;
                    }
                    return glm::vec3(0.0f);
                }

                Ray bounceRay(isect.position + normal * 0.001f, direction);
                return tracePath(bounceRay, scene, depth + 1);
            }
        }

        return glm::vec3(0.0f);
    }

    glm::vec3 calculateDirectLighting(const Intersection& isect, const Scene& scene, const glm::vec3& viewDir) {
        glm::vec3 totalLight(0.0f);
        const auto& lights = scene.getLights();
        const auto& materials = scene.getMaterials();
        const auto& material = materials[isect.materialId];

        for (const auto& light : lights) {
            // Calculate light direction and distance
            glm::vec3 lightDir = light.position - isect.position;
            float lightDistance = glm::length(lightDir);
            
            if (lightDistance < 0.0001f) {
                #pragma omp critical
                {
                    std::cout << "Warning: Light too close to surface" << std::endl;
                }
                continue;
            }

            lightDir = glm::normalize(lightDir);

            // Check for shadows
            Ray shadowRay(isect.position + isect.normal * 0.001f, lightDir);
            shadowRay.tMax = lightDistance - 0.001f;
            Intersection shadowIsect;
            
            if (!scene.intersect(shadowRay, shadowIsect)) {
                // Not in shadow, calculate lighting
                float cosTheta = glm::max(glm::dot(isect.normal, lightDir), 0.0f);
                float attenuation = light.intensity / (lightDistance * lightDistance);

                glm::vec3 brdf;
                if (material->type == MaterialType::DIFFUSE) {
                    brdf = material->albedo / glm::pi<float>();
                } else if (material->type == MaterialType::SPECULAR) {
                    glm::vec3 halfVec = glm::normalize(lightDir + viewDir);
                    float NdotH = glm::max(glm::dot(isect.normal, halfVec), 0.0f);
                    float D = MaterialUtils::ggxDistribution(NdotH, material->roughness);
                    brdf = material->albedo * D;
                }
                
                glm::vec3 contribution = light.color * brdf * cosTheta * attenuation;
                
                if (isValidColor(contribution, "Light contribution")) {
                    totalLight += contribution;
                }
            }
        }
        return totalLight;
    }

    glm::vec3 randomHemisphereDirection(const glm::vec3& normal) {
        glm::vec3 dir = randomInUnitSphere();
        return glm::dot(dir, normal) < 0.0f ? -dir : dir;
    }

    glm::vec3 randomInUnitSphere() {
        while (true) {
            glm::vec3 p = 2.0f * glm::vec3(
                randomFloat(),
                randomFloat(),
                randomFloat()
            ) - glm::vec3(1.0f);
            
            if (glm::dot(p, p) < 1.0f)
                return glm::normalize(p);
        }
    }
};
