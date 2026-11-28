#pragma once

#include <complex>

class GpuFft {
public:
    explicit GpuFft(int n);
    ~GpuFft();
    
    int size() const noexcept { return n_; }

    void forwardDevice(const std::complex<float>* d_in, std::complex<float>* d_out);
    void forward(const std::complex<float>* in, std::complex<float>* out);

private:
    void release() noexcept;

    int n_ = 0;
    std::size_t bytes_ = 0;
    
    std::complex<float>* d_in_ = nullptr;
    std::complex<float>* d_out_ = nullptr;
    std::complex<float>* d_work_[2] = {};
    std::complex<float>* d_rootsUnity_ = nullptr;
};
