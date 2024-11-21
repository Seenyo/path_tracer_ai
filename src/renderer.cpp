#include "../include/renderer.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb/stb_image_write.h"

void Renderer::saveImage(const std::string& filename) {
    std::vector<unsigned char> pixels(settings.width * settings.height * 3);

    for (int i = 0; i < settings.width * settings.height; ++i) {
        // Apply gamma correction and tone mapping
        glm::vec3 color = frameBuffer[i];
        color = glm::pow(glm::clamp(color, 0.0f, 1.0f), glm::vec3(1.0f / settings.gamma));

        // Convert to 8-bit color
        pixels[i * 3 + 0] = static_cast<unsigned char>(color.r * 255.0f);
        pixels[i * 3 + 1] = static_cast<unsigned char>(color.g * 255.0f);
        pixels[i * 3 + 2] = static_cast<unsigned char>(color.b * 255.0f);
    }

    stbi_write_png(filename.c_str(), settings.width, settings.height, 3, pixels.data(), settings.width * 3);
    std::cout << "Image saved as: " << filename << std::endl;
}
