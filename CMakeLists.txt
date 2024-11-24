cmake_minimum_required(VERSION 3.18)
project(path_tracer_ai LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 89)  # For RTX 4090

# Find OptiX SDK
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0")
if(NOT EXISTS ${OptiX_INSTALL_DIR})
    message(FATAL_ERROR "Please install the OptiX SDK to: ${OptiX_INSTALL_DIR}")
endif()

# Set include directories for OptiX and CUDA
set(OptiX_INCLUDE_DIR "${OptiX_INSTALL_DIR}/include")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
message(STATUS "OptiX Include Dir: ${OptiX_INCLUDE_DIR}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")

# Find CUDA libraries
find_library(CUDA_DRIVER_LIBRARY
    NAMES cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
)
find_library(CUDA_RUNTIME_LIBRARY
    NAMES cudart
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
)
message(STATUS "CUDA Driver Library: ${CUDA_DRIVER_LIBRARY}")
message(STATUS "CUDA Runtime Library: ${CUDA_RUNTIME_LIBRARY}")

# Add GLM
add_subdirectory(external/glm)

# Create PTX output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ptx)

# Add CUDA compilation for OptiX shaders
set(OPTIX_SHADER_SOURCES
    src/optix/shaders.cu
)

# Custom command to compile OptiX shaders to PTX
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/ptx/optix_shaders.ptx
    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc
        -ptx
        -arch=sm_89
        -std=c++17
        -O0
        -g
        -Xcompiler /Od
        -D_DEBUG
        -lineinfo
        --use_fast_math
        --relocatable-device-code=true
        -I${OptiX_INCLUDE_DIR}
        -I${CUDA_INCLUDE_DIRS}
        -I${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/optix/shaders.cu
        -o ${CMAKE_BINARY_DIR}/ptx/optix_shaders.ptx
    DEPENDS ${OPTIX_SHADER_SOURCES}
    COMMENT "Compiling ${OPTIX_SHADER_SOURCES} to PTX"
)

# Add custom target for PTX generation
add_custom_target(optix_shaders_ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/ptx/optix_shaders.ptx
)

# Add source files
set(SOURCE_FILES
    src/main.cpp
    src/renderer.cpp
    src/scene.cpp
    src/optix/pipeline.cpp
    src/optix/scene.cpp
)

# Add check_optix executable
add_executable(check_optix src/check_optix.cpp)
target_include_directories(check_optix PRIVATE
    ${OptiX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(check_optix PRIVATE
    ${CUDA_DRIVER_LIBRARY}
    ${CUDA_RUNTIME_LIBRARY}
    OpenMP::OpenMP_CXX
)

# Add main executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Add include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OptiX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CUDA_DRIVER_LIBRARY}
    ${CUDA_RUNTIME_LIBRARY}
    OpenMP::OpenMP_CXX
    glm
)

# Add dependencies
add_dependencies(${PROJECT_NAME} optix_shaders_ptx)

# Set Visual Studio working directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
