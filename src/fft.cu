#include "fft.h"

#include <cuda_runtime.h>

#include <complex>
#include <cmath>
#include <vector>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <string>
#include <cstddef>

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return {a.x * b.x  - a.y * b.y, a.x * b.y + a.y * b.x}; }

float2* asFloat2(std::complex<float>* p) { return reinterpret_cast<float2*>(p); }
const float2* asFloat2(const std::complex<float>* p) { return reinterpret_cast<const float2*>(p); }

// Stockham FFT
__global__ void fftKernel(const float2* __restrict__ in, float2* __restrict__ out, const float2* __restrict__ rootsUnity, const int halfN, const int d){
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

}


GpuFft::GpuFft(int n) : n_(n), bytes_(static_cast<std::size_t>(n) * sizeof(std::complex<float>)) {
    if (n < 1 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("GpuFft: size must be a power of 2, got " + std::to_string(n));
    }

    try {
        check(cudaMalloc(&d_in_, bytes_), "cudaMalloc d_in");
        check(cudaMalloc(&d_out_, bytes_), "cudaMalloc d_out");
        check(cudaMalloc(&d_work_[0], bytes_), "cudaMalloc d_work[0]");
        check(cudaMalloc(&d_work_[1], bytes_), "cudaMalloc d_work[1]");
        check(cudaMalloc(&d_rootsUnity_, bytes_), "cudaMalloc d_rootsUnity");

        std::vector<std::complex<float>> rootsUnity(n);
        for (int i = 0; i < n; i++) {
            const double angle = 2.0 * std::numbers::pi * i / n;
            rootsUnity[i] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(-angle))};
        }
        check(cudaMemcpy(d_rootsUnity_, rootsUnity.data(), bytes_, cudaMemcpyHostToDevice), "cudaMemcpy d_rootsUnity");
    } catch (...) {
        release();
        throw;
    }
}

GpuFft::~GpuFft() { release(); }

void GpuFft::release() noexcept {
    cudaFree(d_rootsUnity_);
    cudaFree(d_work_[0]);
    cudaFree(d_work_[1]);
    cudaFree(d_in_);
    cudaFree(d_out_);
    d_rootsUnity_ = d_work_[0] = d_work_[1] = d_in_ = d_out_ = nullptr;
}

// Asynchronously computes the forward DFT of a device input buffer and writes the result to a device output buffer
void GpuFft::forwardDevice(const std::complex<float>* d_in, std::complex<float>* d_out) {
    if (d_in == d_out) {
        throw std::invalid_argument("GpuFft::forwardDevice: input and output memory addresses must differ");
    }

    if (n_ == 1) {
        check(cudaMemcpy(d_out, d_in, bytes_, cudaMemcpyDeviceToDevice), "cudaMemcpy d_in -> d_out");
        return;
    }

    const int halfN = n_ / 2;
    const int blocks = (halfN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    const float2* src = asFloat2(d_in);
    for (int d = halfN; d >= 1; d >>= 1) {
        float2* dst = (d == 1 ? asFloat2(d_out) : asFloat2(d_work_[1])); 
        fftKernel<<<blocks, THREADS_PER_BLOCK>>>(src, dst, asFloat2(d_rootsUnity_), halfN, d);
        src = dst;
        std::swap(d_work_[0], d_work_[1]);
    }

    check(cudaGetLastError(), "fftKernel launch");
}

// Synchronously computes the forward DFT of a host input buffer and writes the result to a host output buffer
void GpuFft::forward(const std::complex<float>* in, std::complex<float>* out) {
    check(cudaMemcpy(d_in_, in, bytes_, cudaMemcpyHostToDevice), "cudaMemcpy in -> d_in_");
    forwardDevice(d_in_, d_out_);
    check(cudaMemcpy(out, d_out_, bytes_, cudaMemcpyDeviceToHost), "cudaMemcpy d_out_ -> out");
}
