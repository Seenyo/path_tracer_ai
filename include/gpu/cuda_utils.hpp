// cuda_utils.hpp
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <optix.h>

#ifndef __CUDA_ARCH__ // Host code only

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// CUDA Error checking macro for host code
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t error = call;                                   \
        if (error != cudaSuccess)                                   \
        {                                                           \
            std::stringstream ss;                                   \
            ss << "CUDA call (" << #call << ") failed with error: " \
               << cudaGetErrorString(error) << " (" << error << ")" \
               << " at " << __FILE__ << ":" << __LINE__;            \
            throw std::runtime_error(ss.str());                     \
        }                                                           \
    } while (0)

// OptiX Error checking macro for host code
#define OPTIX_CHECK(call)                                            \
    do                                                               \
    {                                                                \
        OptixResult result = call;                                   \
        if (result != OPTIX_SUCCESS)                                 \
        {                                                            \
            std::stringstream ss;                                    \
            ss << "OptiX call (" << #call << ") failed with error: " \
               << optixGetErrorName(result) << " (" << result << ")" \
               << " at " << __FILE__ << ":" << __LINE__;             \
            throw std::runtime_error(ss.str());                      \
        }                                                            \
    } while (0)

#else // Device code

// No-op macros for device code
#define CUDA_CHECK(call)
#define OPTIX_CHECK(call)

#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__ // Host code only

// CUDA Memory Management Template Class
template <typename T>
class CUDABuffer
{
public:
    CUDABuffer();
    ~CUDABuffer();

    // Allocate memory for 'count' elements
    void alloc(size_t count);

    // Allocate and upload data from host
    void alloc_and_upload(const std::vector<T> &data);

    // Upload data from host
    void upload(const T *h_ptr, size_t count);

    // Download data to host
    void download(T *h_ptr, size_t count) const;

    // Free allocated memory
    void free();

    // Get device pointer
    T *get();
    const T *get() const;

    // Get size in bytes
    size_t sizeInBytes() const;

    // Conversion operators
    operator T *() { return d_ptr; }
    operator const T *() const { return d_ptr; }

private:
    T *d_ptr;
    size_t size_bytes;
};

// Include the inline implementations
#include "cuda_utils.inl"

#endif // __CUDA_ARCH__
