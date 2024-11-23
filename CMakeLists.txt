cmake_minimum_required(VERSION 3.15)
project(path_tracer_ai)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add external libraries
add_subdirectory(external/glm)

# Add source files
set(SOURCES
    src/main.cpp
    src/scene.cpp
    src/renderer.cpp
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

# Create executable
add_executable(path_tracer_ai ${SOURCES} ${HEADERS})

# Set include directories for the target
target_include_directories(path_tracer_ai PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
)

# Link libraries
target_link_libraries(path_tracer_ai PRIVATE glm::glm)
if(OpenMP_CXX_FOUND)
    target_link_libraries(path_tracer_ai PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set warning level
if(MSVC)
    target_compile_options(path_tracer_ai PRIVATE /W4)
else()
    target_compile_options(path_tracer_ai PRIVATE -Wall -Wextra -Wpedantic)
endif()