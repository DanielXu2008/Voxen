#include "fft.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>
#include <fstream>

int main() {
    constexpr int N = 4096;                  // Power-of-2 FFT size
    constexpr float sampleRate = 48000.0f;   // Standard 48 kHz audio sample rate
    constexpr float middleC = 261.625565f;   // C4 frequency in Hz

    // 1. Generate a pure Middle C sine wave: x[n] = sin(2 * pi * f * n / fs)
    std::vector<std::complex<float>> signal(N);
    for (int n = 0; n < N; ++n) {
        const float w = 0.5 * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> * n / N)); // Hann window
        const float t = static_cast<float>(n) / sampleRate;
        const float sample = std::sin(2.0f * std::numbers::pi_v<float> * middleC * t);
        signal[n] = {sample * w, 0.0f};
    }

    // 2. Run the GPU FFT
    std::vector<std::complex<float>> spectrum(N);
    GpuFft fft(N);
    fft.forward(signal.data(), spectrum.data());

    // 3. Find the strongest bin in the positive frequencies (0 .. N/2)
    int peakBin = 0;
    float peakMag = 0.0f;
    for (int k = 0; k <= N / 2; ++k) {
        const float mag = std::abs(spectrum[k]);
        if (mag > peakMag) {
            peakMag = mag;
            peakBin = k;
        }
    }

    // Convert peak bin back to Hz: freq = bin * (sampleRate / N)
    const float binWidth = sampleRate / static_cast<float>(N);
    const float detectedFreq = static_cast<float>(peakBin) * binWidth;
    const int expectedBin = static_cast<int>(std::round(middleC / binWidth));

    std::cout << "Expected bin : " << expectedBin << " (" << expectedBin * binWidth << " Hz)\n";
    std::cout << "Detected bin : " << peakBin << " (" << detectedFreq << " Hz)\n";
    std::cout << "Peak magnitude: " << peakMag << " (ideal ~ " << N / 2.0f << ")\n";
    
    std::ofstream csv("spectrum.csv");
    csv << "freq,mag\n";
    for (int k = 0; k <= N / 2; ++k)
        csv << k * sampleRate / N << ',' << std::abs(spectrum[k]) << '\n';

    // 4. Verify the peak is in the right bin and has strong energy
    if (peakBin != expectedBin || peakMag < (N / 2.0f) * 0.5f) {
        std::cerr << "FAIL: Middle C peak mismatch!\n";
        return EXIT_FAILURE;
    }

    std::cout << "PASS!\n";
    return EXIT_SUCCESS;
}
