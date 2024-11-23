#pragma once

#include <cuda_runtime.h>

// CUDA vector operations
__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator*(const float3 &a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float3 operator*(float a, const float3 &b)
{
    return b * a;
}

__device__ __forceinline__ float3 operator/(const float3 &a, float b)
{
    float inv = 1.0f / b;
    return a * inv;
}

__device__ __forceinline__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __forceinline__ float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float3 normalize(const float3 &v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// Random number generation on GPU
__device__ inline float random(unsigned int &seed)
{
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(seed & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

__device__ inline float3 randomInUnitSphere(unsigned int &seed)
{
    while (true)
    {
        float3 p = make_float3(
            2.0f * random(seed) - 1.0f,
            2.0f * random(seed) - 1.0f,
            2.0f * random(seed) - 1.0f);
        if (dot(p, p) < 1.0f)
            return normalize(p);
    }
}

__device__ inline float3 randomHemisphereDirection(const float3 &normal, unsigned int &seed)
{
    float3 dir = randomInUnitSphere(seed);
    return dot(dir, normal) < 0.0f ? -dir : dir;
}

__device__ inline float3 reflect(const float3 &v, const float3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

__device__ inline float3 refract(const float3 &uv, const float3 &n, float etaiOverEtat)
{
    float cosTheta = fminf(dot(-uv, n), 1.0f);
    float3 rOutPerp = etaiOverEtat * (uv + cosTheta * n);
    float3 rOutParallel = -sqrtf(fabsf(1.0f - dot(rOutPerp, rOutPerp))) * n;
    return rOutPerp + rOutParallel;
}
