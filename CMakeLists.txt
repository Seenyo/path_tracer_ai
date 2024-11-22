cmake_minimum_required(VERSION 3.18)
project(path_tracer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Set OptiX paths - adjust these paths according to your OptiX SDK installation
set(OptiX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0" CACHE PATH "Path to OptiX SDK")
set(OptiX_INCLUDE_DIR "${OptiX_ROOT_DIR}/include")

# Find Visual Studio installation
if(MSVC)
    # Get Visual Studio installation path
    get_filename_component(VS_PATH [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VS7;Latest] ABSOLUTE)
    if(NOT VS_PATH)
        get_filename_component(VS_PATH [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VS7;Current] ABSOLUTE)
    endif()
    
    # Set Visual Studio include paths
    set(VS_INCLUDE_PATH "${VS_PATH}/VC/Tools/MSVC/14.32.31326/include")
    set(WIN_SDK_PATH "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0")
endif()

# Find vcpkg packages
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 4090

# Enable separate compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OptiX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${VS_INCLUDE_PATH}
    "${WIN_SDK_PATH}/ucrt"
    "${WIN_SDK_PATH}/um"
    "${WIN_SDK_PATH}/shared"
)

# Source files
set(CPU_SOURCES
    src/main.cpp
    src/renderer.cpp
    src/scene.cpp
)

set(GPU_SOURCES
    src/gpu/optix_renderer.cu
    src/gpu/cuda_utils.cu
)

# PTX compilation
set(PTX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/ptx/optix_kernels.cu")
set(PTX_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/ptx/optix_kernels.ptx")

# Create PTX compilation command using batch script
add_custom_command(
    OUTPUT ${PTX_FILE}
    COMMAND cmd.exe /C "${CMAKE_CURRENT_SOURCE_DIR}/compile_ptx.bat"
    DEPENDS ${PTX_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/compile_ptx.bat
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Compiling OptiX PTX"
    VERBATIM
)

add_custom_target(ptx_target DEPENDS ${PTX_FILE})

# Create executable
add_executable(${PROJECT_NAME} 
    ${CPU_SOURCES}
    ${GPU_SOURCES}
)

add_dependencies(${PROJECT_NAME} ptx_target)

# Apply CUDA compilation flags directly to the target
target_compile_options(${PROJECT_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extended-lambda
        --relocatable-device-code=true
        -lineinfo
        --expt-relaxed-constexpr
        -Xcompiler "/MD"
        --keep-device-functions
    >
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
    )
endif()

# Set CUDA host compiler options
if(MSVC)
    target_compile_options(${PROJECT_NAME}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/MD>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler "/MD">
    )
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
    OpenMP::OpenMP_CXX
    glm::glm
    tinyobjloader::tinyobjloader
    cxxopts::cxxopts
)

# Set output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Create PTX output directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_BINARY_DIR}/bin/ptx
)

# Copy PTX file to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${PTX_FILE}
    ${CMAKE_BINARY_DIR}/bin/ptx/optix_kernels.ptx
)

# Install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install PTX files
install(FILES ${PTX_FILE}
    DESTINATION bin/ptx
)

# Set working directory for Visual Studio
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Set Visual Studio filters
source_group("Source Files\\CPU" FILES ${CPU_SOURCES})
source_group("Source Files\\GPU" FILES ${GPU_SOURCES})
source_group("Source Files\\GPU\\PTX" FILES ${PTX_SOURCE})

# Print configuration summary
message(STATUS "Configuration Summary")
message(STATUS "-------------------")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "OptiX SDK: ${OptiX_ROOT_DIR}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Flags: Applied directly to target")
message(STATUS "Include Dirs:")
message(STATUS "  - ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "  - ${OptiX_INCLUDE_DIR}")
message(STATUS "  - ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  - ${VS_INCLUDE_PATH}")
message(STATUS "  - ${WIN_SDK_PATH}/ucrt")
message(STATUS "  - ${WIN_SDK_PATH}/um")
message(STATUS "  - ${WIN_SDK_PATH}/shared")
