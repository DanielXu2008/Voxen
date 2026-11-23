#include "fft.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <numbers>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return {a.x * b.x  - a.y * b.y, a.x * b.y + a.y * b.x}; }

}

// Stockham FFT
__global__ void fftKernel(const float2* in, float2* out, const float2* rootsUnity, const int halfN, const int d){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= halfN) return;

    const int base = idx & (~(d - 1));
    const float2 w = rootsUnity[base];
    const float2 even = in[base + idx];
    const float2 odd = in[base + idx + d];

    const float2 t = cmul(w, odd);
    out[idx] = even + t;
    out[idx + halfN] = even - t;
}

GpuFft::GpuFft(int n) : n_(n), bytes_(static_cast<std::size_t>(n) * sizeof(std::complex<float>)) {
    cudaMalloc(&d_in_, bytes_);
    cudaMalloc(&d_out_, bytes_);
    cudaMalloc(&d_work_[0], bytes_);
    cudaMalloc(&d_work_[1], bytes_);
    cudaMalloc(&d_rootsUnity_, bytes_);

    std::vector<std::complex<float>> rootsUnity(n);
    for (int i = 0; i < n; i++) {
        const double angle = -2.0 * std::numbers::pi * i / n;
        rootsUnity[i] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(-angle))};
    }
    cudaMemcpy(d_rootsUnity_, rootsUnity.data(), bytes_, cudaMemcpyHostToDevice);
}

GpuFft::~GpuFft() { 
    cudaFree(d_in_);
    cudaFree(d_out_);
    cudaFree(d_work_[0]);
    cudaFree(d_work_[1]);
    cudaFree(d_rootsUnity_);
}

void GpuFft::forwardDevice(std::complex<float>* d_in, std::complex<float>* d_out) {
    if (n_ == 1) {
        cudaMemcpy(d_out, d_in, bytes_, cudaMemcpyDeviceToDevice);
        return;
    }

    const int halfN = n_ / 2;
    const int blocks = (halfN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaMemcpy(d_work_[0], d_in, bytes_, cudaMemcpyDeviceToDevice);
    for (int d = halfN; d >= 1; d >>= 1) {
        fftKernel<<<blocks, THREADS_PER_BLOCK>>>(reinterpret_cast<float2*>(d_work_[0]), reinterpret_cast<float2*>(d_work_[1]), reinterpret_cast<float2*>(d_rootsUnity_), halfN, d);
        std::swap(d_work_[0], d_work_[1]);
    }
    cudaMemcpy(d_out, d_work_[0], bytes_, cudaMemcpyDeviceToDevice);
}

void GpuFft::forward(std::complex<float>* in, std::complex<float>* out) {
    cudaMemcpy(d_in_, in, bytes_, cudaMemcpyHostToDevice);
    forwardDevice(d_in_, d_out_);
    cudaMemcpy(out, d_out_, bytes_, cudaMemcpyDeviceToHost);
}
