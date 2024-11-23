cmake_minimum_required(VERSION 3.15)
project(path_tracer_ai LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# -----------------------------------
# CUDA Configuration
# -----------------------------------

# Find CUDA
find_package(CUDA REQUIRED)

# Verify CUDA is found
if (CUDA_FOUND)
    message(STATUS "CUDA toolkit found at: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(FATAL_ERROR "CUDA toolkit not found. Please install CUDA.")
endif()

# Optionally set CUDA architecture (adjust based on your GPU)
# set(CUDA_ARCHITECTURES 75)

# Add CUDA specific compile options if needed
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Add external libraries
add_subdirectory(external/glm)

# Add source files
set(SOURCES
        src/main.cpp
        src/scene.cpp
        src/renderer.cpp
        # Add CUDA source files here if any, e.g.,
        # src/kernel.cu
        )

# Add header files
set(HEADERS
        include/camera.hpp
        include/scene.hpp
        include/material.hpp
        include/ray.hpp
        include/intersection.hpp
        include/renderer.hpp
        include/bvh.hpp
        )

# -----------------------------------
# OptiX Configuration
# -----------------------------------

# Set the OptiX SDK directory
set(OPTIX_SDK_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0" CACHE PATH "Path to OptiX SDK")

# Define include and library directories
set(OPTIX_INCLUDE_DIR "${OPTIX_SDK_DIR}/include")

# Find OptiX header
find_path(OPTIX_INCLUDE_DIR_PATH optix.h HINTS "${OPTIX_INCLUDE_DIR}")


# Check if OptiX was found
if(NOT OPTIX_INCLUDE_DIR_PATH)
    message(FATAL_ERROR "OptiX include directory not found. Please verify OPTIX_SDK_DIR.")
endif()

# Include OptiX headers as system includes to suppress warnings from OptiX
# (Note: Already handled in target_include_directories below)

# -----------------------------------
# Create Main Executable
# -----------------------------------

# Create executable for the main application
add_executable(path_tracer_ai ${SOURCES} ${HEADERS})

# Set include directories for the main target
target_include_directories(path_tracer_ai PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
        ${CUDA_INCLUDE_DIRS}
        )

# Include CUDA and OptiX system directories
target_include_directories(path_tracer_ai SYSTEM PRIVATE
        ${OPTIX_INCLUDE_DIR_PATH}
        ${CUDA_INCLUDE_DIRS}
        )

# Define compile definitions to suppress deprecated warnings
if(MSVC)
    target_compile_definitions(path_tracer_ai PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Enable CUDA in the target
set_target_properties(path_tracer_ai PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        )

# Link libraries for the main target
target_link_libraries(path_tracer_ai PRIVATE glm::glm)
if(OpenMP_CXX_FOUND)
    target_link_libraries(path_tracer_ai PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(path_tracer_ai PRIVATE ${CUDA_LIBRARIES})

# Suppress specific MSVC warnings
if(MSVC)
    target_compile_options(path_tracer_ai PRIVATE
            /W4
            /wd4819   # Suppress C4819
            /wd4267   # Suppress C4267
            /wd4996   # Suppress C4996
            )
    target_compile_options(path_tracer_ai PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:/W4 /wd4819 /wd4267 /wd4996>)
else()
    target_compile_options(path_tracer_ai PRIVATE
            -Wall -Wextra -Wpedantic
            # Remove -Werror to prevent warnings from being treated as errors
            )
    target_compile_options(path_tracer_ai PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra -Wpedantic>)
    # Optionally, add flags to disable specific warnings if needed
endif()

# -----------------------------------
# Create Check OptiX Executable
# -----------------------------------

# Add source for OptiX check
set(CHECK_OPTIX_SOURCES
        src/check_optix.cpp
        )

# Create executable to check OptiX availability
add_executable(check_optix ${CHECK_OPTIX_SOURCES})

# Set include directories for the check_optix target
target_include_directories(check_optix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
        ${CUDA_INCLUDE_DIRS}
        )

# Include CUDA and OptiX system directories
target_include_directories(check_optix SYSTEM PRIVATE
        ${OPTIX_INCLUDE_DIR_PATH}
        ${CUDA_INCLUDE_DIRS}
        )

# Define compile definitions to suppress deprecated warnings
if(MSVC)
    target_compile_definitions(check_optix PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Link libraries for the check_optix target
target_link_libraries(check_optix PRIVATE glm::glm)
if(OpenMP_CXX_FOUND)
    target_link_libraries(check_optix PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(check_optix PRIVATE ${CUDA_LIBRARIES})

# Suppress specific MSVC warnings
if(MSVC)
    target_compile_options(check_optix PRIVATE
            /W4
            /wd4819   # Suppress C4819
            /wd4267   # Suppress C4267
            /wd4996   # Suppress C4996
            )
    target_compile_options(check_optix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:/W4 /wd4819 /wd4267 /wd4996>)
else()
    target_compile_options(check_optix PRIVATE
            -Wall -Wextra -Wpedantic
            # Remove -Werror to prevent warnings from being treated as errors
            )
    target_compile_options(check_optix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra -Wpedantic>)
    # Optionally, add flags to disable specific warnings if needed
endif()
