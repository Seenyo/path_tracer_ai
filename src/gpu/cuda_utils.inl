#pragma once

template <typename T>
CUDABuffer<T>::CUDABuffer()
    : d_ptr(nullptr), alloc_size(0) {}

template <typename T>
CUDABuffer<T>::~CUDABuffer()
{
    free();
}

template <typename T>
void CUDABuffer<T>::alloc(size_t size)
{
    if (d_ptr)
    {
        free();
    }
    alloc_size = size * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d_ptr, alloc_size));
}

template <typename T>
void CUDABuffer<T>::alloc_and_upload(const std::vector<T> &data)
{
    alloc(data.size());
    upload(data.data(), data.size());
}

template <typename T>
void CUDABuffer<T>::upload(const T *data, size_t size)
{
    if (size * sizeof(T) > alloc_size)
    {
        throw std::runtime_error("CUDABuffer upload size exceeds allocated size");
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, data, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void CUDABuffer<T>::download(T *data, size_t size)
{
    if (size * sizeof(T) > alloc_size)
    {
        throw std::runtime_error("CUDABuffer download size exceeds allocated size");
    }
    CUDA_CHECK(cudaMemcpy(data, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void CUDABuffer<T>::free()
{
    if (d_ptr)
    {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr = nullptr;
        alloc_size = 0;
    }
}

template <typename T>
T *CUDABuffer<T>::get() const
{
    return d_ptr;
}

template <typename T>
T *CUDABuffer<T>::d_pointer() const
{
    return d_ptr;
}
