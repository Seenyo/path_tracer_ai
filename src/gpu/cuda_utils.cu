// cuda_utils.cu

#include "../../include/gpu/cuda_utils.hpp"

#include <iostream>
#include <stdexcept>

// Definitions of utility functions

void initializeCUDAErrorHandling() {
    // Currently, CUDA does not support setting a custom error handler directly.
    // You can use cudaDeviceSetLimit or other mechanisms if needed.
    // For now, we'll leave this function empty.
}

void printDeviceProperties() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));

        std::cout << "\nCUDA Device " << device << ": " << props.name << std::endl;
        std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "  Total Global Memory: "
                  << props.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: "
                  << props.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Number of Multiprocessors: " << props.multiProcessorCount << std::endl;
        std::cout << "  Warp Size: " << props.warpSize << std::endl;
        std::cout << "  Memory Clock Rate: "
                  << props.memoryClockRate / 1000.0 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << props.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth: "
                  << 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6
                  << " GB/s" << std::endl;
    }
}

void* allocateDeviceMemory(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDeviceMemory(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

cudaStream_t createCUDAStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

void destroyCUDAStream(cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void synchronizeDevice() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void copyToDevice(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void copyToHost(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

cudaEvent_t createCUDAEvent() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

void destroyCUDAEvent(cudaEvent_t event) {
    if (event) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
}

float getEventElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

// Optionally, you can include any other non-inline CUDA utility functions here.
// For example, device synchronization, error handling, etc.

