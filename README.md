# GPU-Accelerated Path Tracer

A physically-based path tracer with both CPU and GPU rendering capabilities, utilizing NVIDIA OptiX for GPU acceleration.

## Features

- Physically-based path tracing
- GPU acceleration using NVIDIA OptiX
- Support for multiple material types:
  - Diffuse
  - Specular
  - Dielectric (glass/transparent materials)
- OBJ file loading with material support
- Configurable rendering settings
- Automatic fallback to CPU rendering if GPU rendering fails

## Requirements

- Windows 11 (64-bit)
- NVIDIA GPU with RTX capabilities (RTX 4090 recommended)
- CUDA Toolkit 12.3
- OptiX SDK 7.7.0
- Visual Studio 2022
- CMake 3.18 or higher
- vcpkg package manager

## Dependencies

The project uses the following libraries:
- GLM (OpenGL Mathematics)
- TinyObjLoader
- cxxopts
- stb_image_write

## Installation

1. Install the required software:
   - [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
   - [CUDA Toolkit 12.3](https://developer.nvidia.com/cuda-downloads)
   - [OptiX SDK 7.7.0](https://developer.nvidia.com/designworks/optix/downloads/7.7.0/windows)
   - [CMake](https://cmake.org/download/)
   - [vcpkg](https://github.com/Microsoft/vcpkg)

2. Clone the repository:
   ```powershell
   git clone https://github.com/yourusername/path_tracer.git
   cd path_tracer
   ```

3. Update the paths in `build.ps1` to match your system:
   - `$cudaPath`: Path to CUDA Toolkit installation
   - `$optixPath`: Path to OptiX SDK installation
   - `$vcpkgPath`: Path to vcpkg installation

4. Run the build script as Administrator:
   ```powershell
   powershell -ExecutionPolicy Bypass -File build.ps1
   ```

## Usage

The path tracer can be run in either CPU or GPU mode with various configuration options:

```powershell
# GPU rendering (recommended)
.\build\bin\Release\path_tracer.exe --mode gpu --width 1920 --height 1080 --samples 100

# CPU rendering
.\build\bin\Release\path_tracer.exe --mode cpu --width 800 --height 450 --samples 10
```

### Command Line Options

- `--mode, -m`: Rendering mode (cpu/gpu) [default: gpu]
- `--width, -w`: Image width [default: 800]
- `--height, -h`: Image height [default: 450]
- `--samples, -s`: Samples per pixel [default: 100]
- `--bounces, -b`: Maximum ray bounces [default: 5]
- `--gamma, -g`: Gamma correction value [default: 2.2]
- `--input, -i`: Input OBJ file path [default: "IronMan/IronMan.obj"]
- `--output, -o`: Output image file path [default: "output.png"]
- `--help`: Print help message

## Project Structure

```
path_tracer/
├── include/
│   ├── gpu/
│   │   ├── cuda_utils.hpp
│   │   ├── optix_renderer.hpp
│   │   └── optix_types.hpp
│   ├── aabb.hpp
│   ├── bvh.hpp
│   ├── camera.hpp
│   ├── intersection.hpp
│   ├── material.hpp
│   ├── ray.hpp
│   ├── renderer.hpp
│   ├── scene.hpp
│   └── triangle.hpp
├── src/
│   ├── gpu/
│   │   ├── ptx/
│   │   │   └── optix_kernels.cu
│   │   └── optix_renderer.cu
│   ├── main.cpp
│   ├── renderer.cpp
│   └── scene.cpp
├── build.ps1
├── CMakeLists.txt
└── README.md
```

## Performance

The GPU-accelerated version using OptiX can achieve significant speedups compared to the CPU version:

- RTX 4090 (GPU): ~10-100x faster than CPU rendering
- Performance varies based on scene complexity, sample count, and resolution

## Implementation Details

### GPU Acceleration

The GPU implementation uses NVIDIA OptiX for ray tracing acceleration:
- Custom OptiX programs for ray generation, closest hit, any hit, and miss
- Hardware-accelerated BVH traversal
- Efficient parallel computation of path tracing

### Materials

Supports three material types:
1. Diffuse: Lambertian diffuse reflection
2. Specular: Perfect and rough specular reflection
3. Dielectric: Glass-like materials with refraction

### Lighting

- Direct lighting with multiple light sources
- Soft shadows
- Global illumination through path tracing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for OptiX and CUDA
- The path tracing community for algorithms and techniques
- Contributors to the open-source libraries used in this project
