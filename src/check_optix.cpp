#define OPTIX_API_IMPORT
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <optix_stubs.h>

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    } else {
        std::cout << "CUDA initialized successfully." << std::endl;
    }

    // Initialize OptiX
    OptixResult optixStatus = optixInit();
    if (optixStatus == OPTIX_SUCCESS) {
        std::cout << "OptiX is available and initialized successfully." << std::endl;
        return 0;
    } else {
        std::cerr << "Failed to initialize OptiX. Error code: " << optixStatus << std::endl;
        return 1;
    }
}
