#include "../include/scene.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "../external/tinyobjloader/tiny_obj_loader.h"
#include <iostream>

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

    std::cout << "Model statistics:" << std::endl;
    std::cout << "- Vertices: " << attrib.vertices.size() / 3 << std::endl;
    std::cout << "- Normals: " << attrib.normals.size() / 3 << std::endl;
    std::cout << "- TexCoords: " << attrib.texcoords.size() / 2 << std::endl;
    std::cout << "- Shapes: " << shapes.size() << std::endl;
    std::cout << "- Materials: " << objMaterials.size() << std::endl;

    // Create a default red metallic material for Iron Man
    std::cout << "Creating default material..." << std::endl;
    auto defaultMaterial = std::make_shared<Material>();
    defaultMaterial->type = MaterialType::SPECULAR;
    defaultMaterial->albedo = glm::vec3(0.8f, 0.2f, 0.2f);  // Red color
    defaultMaterial->roughness = 0.2f;  // Fairly smooth
    defaultMaterial->metallic = 0.8f;   // Metallic
    materials.push_back(defaultMaterial);

    // Process materials from MTL file
    std::cout << "Processing materials..." << std::endl;
    for (const auto& material : objMaterials) {
        auto mat = std::make_shared<Material>();
        
        // Determine material type based on properties
        if (material.illum == 5 || material.illum == 7) {
            mat->type = MaterialType::SPECULAR;
            mat->roughness = material.roughness;
            mat->metallic = 1.0f;
        } else if (material.illum == 7) {
            mat->type = MaterialType::DIELECTRIC;
            mat->ior = material.ior;
            mat->roughness = material.roughness;
        } else {
            mat->type = MaterialType::DIFFUSE;
            mat->roughness = material.roughness;
        }
        
        mat->albedo = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
        materials.push_back(mat);
        
        std::cout << "Loaded material: " << material.name 
                 << " (type=" << static_cast<int>(mat->type) 
                 << ", roughness=" << mat->roughness << ")" << std::endl;
    }

    // Clear existing triangles before loading new ones
    triangles.clear();

    // Process shapes
    std::cout << "Processing shapes..." << std::endl;
    int totalTriangles = 0;
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                std::cout << "Warning: Skipping non-triangular face" << std::endl;
                continue;
            }

            glm::vec3 vertices[3];
            glm::vec3 normals[3];
            glm::vec2 uvs[3];

            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                
                // Vertex position
                vertices[v] = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Normal
                if (idx.normal_index >= 0) {
                    normals[v] = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                } else {
                    // Calculate face normal if not provided
                    if (v == 2) {
                        glm::vec3 edge1 = vertices[1] - vertices[0];
                        glm::vec3 edge2 = vertices[2] - vertices[0];
                        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
                        normals[0] = normals[1] = normals[2] = normal;
                    }
                }

                // Texture coordinates
                if (idx.texcoord_index >= 0) {
                    uvs[v] = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1]
                    );
                } else {
                    uvs[v] = glm::vec2(0.0f);
                }
            }

            // Use default material (index 0) if no material is specified
            int materialId = shape.mesh.material_ids[f];
            if (materialId < 0) materialId = 0;

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
