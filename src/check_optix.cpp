#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_runtime.h>

// OptiX 7 dependencies (order matters)
#define OPTIX_API_IMPORT
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <optix_function_table_definition.h>

#include <iostream>
#include <sstream>
#include <vector>

// Error check/report helper for CUDA
#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t error = call;                                               \
        if (error != cudaSuccess) {                                             \
            std::stringstream ss;                                               \
            ss << "CUDA call (" << #call << " ) failed with error: '"           \
               << cudaGetErrorString(error) << "' (" __FILE__ << ":"            \
               << __LINE__ << ")\n";                                            \
            std::cerr << ss.str();                                             \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Error check/report helper for OptiX
#define OPTIX_CHECK(call)                                                        \
    do {                                                                         \
        OptixResult res = call;                                                 \
        if (res != OPTIX_SUCCESS) {                                            \
            std::stringstream ss;                                               \
            ss << "OptiX call (" << #call << " ) failed with error: '"         \
               << optixGetErrorName(res) << "' (" __FILE__ << ":"              \
               << __LINE__ << ")\n";                                            \
            std::cerr << ss.str();                                             \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Callback function for OptiX errors
static void context_log_cb(unsigned int level,
                          const char* tag,
                          const char* message,
                          void* /*cbdata */) {
    std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
}

int main() {
    try {
        // Initialize CUDA
        std::cout << "Initializing CUDA..." << std::endl;
        CUDA_CHECK(cudaFree(0));  // Initialize CUDA context
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        // Get and print device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using CUDA Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

        // Initialize OptiX
        std::cout << "\nInitializing OptiX..." << std::endl;
        OPTIX_CHECK(optixInit());

        // Create context
        OptixDeviceContext context = nullptr;
        CUcontext cuCtx = 0;  // zero means take the current context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;  // Fatal = 1, Error = 2, Warning = 3, Print = 4
        
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

        // Configure context
        OPTIX_CHECK(optixDeviceContextSetLogCallback(context, context_log_cb, nullptr, 4));

        std::cout << "\nOptiX and CUDA Setup Summary:" << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "OptiX Context: Successfully created" << std::endl;
        std::cout << "--------------------------------" << std::endl;

        // Clean up
        if (context) {
            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
