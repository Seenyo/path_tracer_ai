#include "../include/scene.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../external/tinyobjloader/tiny_obj_loader.h"
#include <iostream>
#include <limits>
#include <glm/glm.hpp>

bool Scene::loadFromObj(const std::string& objPath) {
    std::cout << "Loading model from: " << objPath << std::endl;
    
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    if (!reader.ParseFromFile(objPath, config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader error: " << reader.Error();
        }
        return false;
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader warning: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& objMaterials = reader.GetMaterials();

    // Calculate model bounds
    glm::vec3 minBounds(std::numeric_limits<float>::max());
    glm::vec3 maxBounds(-std::numeric_limits<float>::max());
    
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        glm::vec3 vertex(
            attrib.vertices[i],
            attrib.vertices[i + 1],
            attrib.vertices[i + 2]
        );
        minBounds = glm::min(minBounds, vertex);
        maxBounds = glm::max(maxBounds, vertex);
    }
    
    glm::vec3 modelSize = maxBounds - minBounds;
    std::cout << "Original model size: " << modelSize.x << ", " << modelSize.y << ", " << modelSize.z << std::endl;
    
    // Calculate scaling factor to fit model in a 1.5-unit box
    float targetSize = 3.f;
    float scaleFactor = targetSize / glm::max(glm::max(modelSize.x, modelSize.y), modelSize.z);
    
    // Calculate center offset to place model at origin
    glm::vec3 centerOffset = (minBounds + maxBounds) * 0.5f;

    // Create materials
    std::cout << "Creating materials..." << std::endl;
    auto defaultMaterial = std::make_shared<Material>();
    defaultMaterial->type = MaterialType::SPECULAR;
    defaultMaterial->albedo = glm::vec3(0.8f, 0.2f, 0.2f);
    defaultMaterial->roughness = 0.2f;
    defaultMaterial->metallic = 0.8f;
    materials.push_back(defaultMaterial);

    // Create wall material
    auto wallMaterial = std::make_shared<Material>();
    wallMaterial->type = MaterialType::DIFFUSE;
    wallMaterial->albedo = glm::vec3(0.9f, 0.9f, 0.9f);
    wallMaterial->roughness = 0.95f;
    wallMaterial->metallic = 0.0f;
    materials.push_back(wallMaterial);

    // Process materials from MTL file
    for (const auto& material : objMaterials) {
        auto mat = std::make_shared<Material>();
        
        if (material.illum == 3) {
            mat->type = MaterialType::DIELECTRIC;
            mat->ior = material.ior > 0.0f ? material.ior : 1.5f;
            mat->roughness = material.roughness > 0.0f ? material.roughness : 0.1f;
        } else if (material.specular[0] > 0.5f || material.specular[1] > 0.5f || material.specular[2] > 0.5f) {
            mat->type = MaterialType::SPECULAR;
            mat->roughness = material.roughness > 0.0f ? material.roughness : 0.3f;
            mat->metallic = material.metallic > 0.0f ? material.metallic : 0.8f;
        } else {
            mat->type = MaterialType::DIFFUSE;
            mat->roughness = material.roughness > 0.0f ? material.roughness : 0.9f;
            mat->metallic = 0.0f;
        }

        mat->albedo = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
        materials.push_back(mat);
    }

    triangles.clear();

    // Add room walls
    const float roomSize = 4.0f;
    const float roomHeight = 3.0f;
    const int wallMatId = 1;

    // Floor
    triangles.emplace_back(
        glm::vec3(-roomSize, 0.0f, -roomSize),
        glm::vec3(roomSize, 0.0f, -roomSize),
        glm::vec3(roomSize, 0.0f, roomSize),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec2(0.0f), glm::vec2(1.0f, 0.0f), glm::vec2(1.0f),
        wallMatId
    );
    triangles.emplace_back(
        glm::vec3(-roomSize, 0.0f, -roomSize),
        glm::vec3(roomSize, 0.0f, roomSize),
        glm::vec3(-roomSize, 0.0f, roomSize),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec2(0.0f), glm::vec2(1.0f), glm::vec2(0.0f, 1.0f),
        wallMatId
    );

    // Back wall
    triangles.emplace_back(
        glm::vec3(-roomSize, 0.0f, -roomSize),
        glm::vec3(-roomSize, roomHeight, -roomSize),
        glm::vec3(roomSize, roomHeight, -roomSize),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec2(0.0f), glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 1.0f),
        wallMatId
    );
    triangles.emplace_back(
        glm::vec3(-roomSize, 0.0f, -roomSize),
        glm::vec3(roomSize, roomHeight, -roomSize),
        glm::vec3(roomSize, 0.0f, -roomSize),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec2(0.0f), glm::vec2(1.0f, 1.0f), glm::vec2(1.0f, 0.0f),
        wallMatId
    );

    // Left wall
    triangles.emplace_back(
        glm::vec3(-roomSize, 0.0f, -roomSize),
        glm::vec3(-roomSize, 0.0f, roomSize),
        glm::vec3(-roomSize, roomHeight, roomSize),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec2(0.0f), glm::vec2(1.0f, 0.0f), glm::vec2(1.0f, 1.0f),
        wallMatId
    );
    triangles.emplace_back(
        glm::vec3(-roomSize, 0.0f, -roomSize),
        glm::vec3(-roomSize, roomHeight, roomSize),
        glm::vec3(-roomSize, roomHeight, -roomSize),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec2(0.0f), glm::vec2(1.0f, 1.0f), glm::vec2(0.0f, 1.0f),
        wallMatId
    );

    // Right wall
    triangles.emplace_back(
        glm::vec3(roomSize, 0.0f, -roomSize),
        glm::vec3(roomSize, roomHeight, roomSize),
        glm::vec3(roomSize, 0.0f, roomSize),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec2(0.0f), glm::vec2(1.0f, 1.0f), glm::vec2(1.0f, 0.0f),
        wallMatId
    );
    triangles.emplace_back(
        glm::vec3(roomSize, 0.0f, -roomSize),
        glm::vec3(roomSize, roomHeight, -roomSize),
        glm::vec3(roomSize, roomHeight, roomSize),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec2(0.0f), glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 1.0f),
        wallMatId
    );

    // Process model triangles
    std::cout << "Processing shapes..." << std::endl;
    int totalTriangles = triangles.size();
    
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) continue;

            glm::vec3 vertices[3];
            glm::vec3 normals[3];
            glm::vec2 uvs[3];

            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                
                // Get vertex position
                glm::vec3 vertex(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Apply transformations manually
                vertex = (vertex - centerOffset) * scaleFactor;  // Center and scale
                vertex.z = -vertex.z;  // Rotate 180 degrees around Y axis
                vertex.y += 0.8f;  // Move up

                vertices[v] = vertex;

                // Handle normal
                if (idx.normal_index >= 0) {
                    glm::vec3 normal(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                    normal.z = -normal.z;  // Rotate normal 180 degrees around Y axis
                    normals[v] = glm::normalize(normal);
                } else if (v == 2) {
                    glm::vec3 edge1 = vertices[1] - vertices[0];
                    glm::vec3 edge2 = vertices[2] - vertices[0];
                    glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
                    normals[0] = normals[1] = normals[2] = normal;
                }

                if (idx.texcoord_index >= 0) {
                    uvs[v] = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1]
                    );
                } else {
                    uvs[v] = glm::vec2(0.0f);
                }
            }

            int materialId = shape.mesh.material_ids[f];
            if (materialId < 0) materialId = 0;
            materialId += 2;  // Skip default material and wall material

            triangles.emplace_back(
                vertices[0], vertices[1], vertices[2],
                normals[0], normals[1], normals[2],
                uvs[0], uvs[1], uvs[2],
                materialId
            );
            
            totalTriangles++;
            index_offset += fv;
        }
    }

    std::cout << "Model loaded successfully:" << std::endl;
    std::cout << "- Total triangles: " << totalTriangles << std::endl;
    std::cout << "- Total materials: " << materials.size() << std::endl;

    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    bvh.build(triangles);

    return true;
}
