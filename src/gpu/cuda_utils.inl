// cuda_utils.inl
#ifndef CUDA_UTILS_INL
#define CUDA_UTILS_INL

#include "../../include/gpu/cuda_utils.hpp"

#ifndef __CUDA_ARCH__ // Host code only

// Constructor
template <typename T>
CUDABuffer<T>::CUDABuffer()
    : d_ptr(nullptr), size_bytes(0) {}

// Destructor
template <typename T>
CUDABuffer<T>::~CUDABuffer()
{
    try
    {
        free();
    }
    catch (const std::exception &e)
    {
        // Handle exceptions in destructor gracefully
        std::cerr << "Exception in CUDABuffer destructor: " << e.what() << std::endl;
    }
}

// Allocate memory for 'count' elements
template <typename T>
void CUDABuffer<T>::alloc(size_t count)
{
    free(); // Free existing memory if allocated
    size_bytes = count * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d_ptr, size_bytes));
}

// Allocate and upload data from host
template <typename T>
void CUDABuffer<T>::alloc_and_upload(const std::vector<T> &data)
{
    alloc(data.size());
    upload(data.data(), data.size());
}

// Upload data from host
template <typename T>
void CUDABuffer<T>::upload(const T *h_ptr, size_t count)
{
    if (!d_ptr)
    {
        throw std::runtime_error("CUDABuffer upload called on unallocated buffer.");
    }

    size_t required_size = count * sizeof(T);
    if (size_bytes < required_size)
    {
        throw std::runtime_error("CUDABuffer upload size exceeds allocated size.");
    }

    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, required_size, cudaMemcpyHostToDevice));
}

// Download data to host
template <typename T>
void CUDABuffer<T>::download(T *h_ptr, size_t count) const
{
    if (!d_ptr)
    {
        throw std::runtime_error("CUDABuffer download called on unallocated buffer.");
    }

    size_t required_size = count * sizeof(T);
    if (size_bytes < required_size)
    {
        throw std::runtime_error("CUDABuffer download size exceeds allocated size.");
    }

    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, required_size, cudaMemcpyDeviceToHost));
}

// Free allocated memory
template <typename T>
void CUDABuffer<T>::free()
{
    if (d_ptr)
    {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr = nullptr;
        size_bytes = 0;
    }
}

// Get device pointer
template <typename T>
T *CUDABuffer<T>::get()
{
    return d_ptr;
}

template <typename T>
const T *CUDABuffer<T>::get() const
{
    return d_ptr;
}

// Get size in bytes
template <typename T>
size_t CUDABuffer<T>::sizeInBytes() const
{
    return size_bytes;
}

#endif // __CUDA_ARCH__

#endif // CUDA_UTILS_INL
