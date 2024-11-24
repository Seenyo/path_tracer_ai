#pragma once

#include <vector>
#include <string>
#include <fstream>

// Simple PPM writer
inline void save_ppm(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), width * height * 3);
}

// Simple PNG writer using raw RGBA data
inline void save_png(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    // First save as PPM, then use an external tool to convert to PNG
    save_ppm(filename + ".ppm", data, width, height);
}
