#pragma once

#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Vector operators
__forceinline__ __device__ __host__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ __host__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ __host__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __device__ __host__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ __host__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ __host__ float3 operator*(float a, const float3& b) {
    return b * a;
}

__forceinline__ __device__ __host__ float3 operator/(const float3& a, float b) {
    float inv = 1.0f / b;
    return a * inv;
}

__forceinline__ __device__ __host__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__forceinline__ __device__ __host__ float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__forceinline__ __device__ __host__ float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

__forceinline__ __device__ __host__ float3& operator/=(float3& a, float b) {
    float inv = 1.0f / b;
    a *= inv;
    return a;
}

__forceinline__ __device__ __host__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ __host__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__forceinline__ __device__ __host__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__forceinline__ __device__ __host__ float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

__forceinline__ __device__ __host__ float3 lerp(const float3& a, const float3& b, float t) {
    return a + t * (b - a);
}

__forceinline__ __device__ __host__ float3 make_float3(float x) {
    return make_float3(x, x, x);
}

__forceinline__ __device__ __host__ float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__forceinline__ __device__ __host__ bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
    float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
    return false;
}

__forceinline__ __device__ __host__ float3 faceforward(const float3& n, const float3& i) {
    return dot(n, i) < 0.0f ? n : -n;
}

__forceinline__ __device__ __host__ float3 fminf(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__forceinline__ __device__ __host__ float3 fmaxf(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__forceinline__ __device__ __host__ float3 clamp(const float3& v, float a, float b) {
    return make_float3(
        fmaxf(a, fminf(b, v.x)),
        fmaxf(a, fminf(b, v.y)),
        fmaxf(a, fminf(b, v.z))
    );
}