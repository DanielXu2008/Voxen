#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Runs on the GPU. Each thread adds one element.
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {            // the last block may have extra threads
        c[i] = a[i] + b[i];
    }
}

int main() {
    // ---- Part A: GPU info ----
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s  CC %d.%d  SMs: %d  Mem: %zu MB\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.totalGlobalMem >> 20);

    // ---- Part B: vector add ----
    const int n = 1 << 20;                  // 1,048,576 elements
    const size_t bytes = n * sizeof(float);

    // 1. Host (CPU) arrays
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * i;
    }

    // 2. Device (GPU) arrays
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    // 3. Copy inputs CPU -> GPU
    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    // 4. Launch: enough blocks of 256 threads to cover n
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;   // round up
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);

    // 5. Error checks
    cudaError_t err = cudaGetLastError();             // was the launch valid?
    printf("Launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();                    // did the kernel run OK?
    printf("Sync:   %s\n", cudaGetErrorString(err));

    // 6. Copy result GPU -> CPU
    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

    // 7. Verify on the CPU
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-3f) {
            if (errors < 5) printf("Mismatch at %d: got %f, expected %f\n", i, c[i], expected);
            ++errors;
        }
    }
    printf(errors == 0 ? "PASS\n" : "FAIL: %d errors\n", errors);

    // 8. Free GPU memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return errors == 0 ? 0 : 1;
}
