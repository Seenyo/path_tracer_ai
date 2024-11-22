#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA Error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t error = call;                                               \
        if (error != cudaSuccess) {                                             \
            std::stringstream ss;                                               \
            ss << "CUDA call (" << #call << ") failed with error: "             \
               << cudaGetErrorString(error) << " (" << error << ")"             \
               << " at " << __FILE__ << ":" << __LINE__;                        \
            throw std::runtime_error(ss.str());                                 \
        }                                                                       \
    } while (0)

// OptiX Error checking
#define OPTIX_CHECK(call)                                                        \
    do {                                                                         \
        OptixResult result = call;                                              \
        if (result != OPTIX_SUCCESS) {                                          \
            std::stringstream ss;                                               \
            ss << "OptiX call (" << #call << ") failed with error: "           \
               << optixGetErrorName(result) << " (" << result << ")"           \
               << " at " << __FILE__ << ":" << __LINE__;                       \
            throw std::runtime_error(ss.str());                                 \
        }                                                                       \
    } while (0)

// CUDA vector operations
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator*(float a, const float3& b) {
    return b * a;
}

__device__ __forceinline__ float3 operator/(const float3& a, float b) {
    float inv = 1.0f / b;
    return a * inv;
}

__device__ __forceinline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// CUDA Memory Management
template<typename T>
class CUDABuffer {
public:
    CUDABuffer() : d_ptr(nullptr), size_bytes(0) {}
    
    ~CUDABuffer() {
        free();
    }

    void alloc(size_t count) {
        free();
        size_bytes = count * sizeof(T);
        CUDA_CHECK(cudaMalloc(&d_ptr, size_bytes));
    }

    void free() {
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
            size_bytes = 0;
        }
    }

    void upload(const T* h_ptr, size_t count) {
        if (size_bytes < count * sizeof(T)) {
            alloc(count);
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T* h_ptr, size_t count) const {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }
    size_t sizeInBytes() const { return size_bytes; }

    operator T*() { return d_ptr; }
    operator const T*() const { return d_ptr; }

private:
    T* d_ptr;
    size_t size_bytes;
};

// Random number generation on GPU
__device__ inline float random(unsigned int& seed) {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

__device__ inline float3 randomInUnitSphere(unsigned int& seed) {
    while (true) {
        float3 p = make_float3(
            2.0f * random(seed) - 1.0f,
            2.0f * random(seed) - 1.0f,
            2.0f * random(seed) - 1.0f
        );
        if (dot(p, p) < 1.0f)
            return normalize(p);
    }
}

__device__ inline float3 randomHemisphereDirection(const float3& normal, unsigned int& seed) {
    float3 dir = randomInUnitSphere(seed);
    return dot(dir, normal) < 0.0f ? -dir : dir;
}

__device__ inline float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ inline float3 refract(const float3& uv, const float3& n, float etaiOverEtat) {
    float cosTheta = fminf(dot(-uv, n), 1.0f);
    float3 rOutPerp = etaiOverEtat * (uv + cosTheta * n);
    float3 rOutParallel = -sqrtf(fabsf(1.0f - dot(rOutPerp, rOutPerp))) * n;
    return rOutPerp + rOutParallel;
}
