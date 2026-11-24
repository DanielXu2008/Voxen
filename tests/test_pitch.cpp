#include "fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace {

constexpr double SAMPLE_RATE = 48000.0;

// Collects `windowLen` samples, applies a `windowLen`-point Hann window,
// zero-pads out to `fft.size()`, and returns the estimated frequency in Hz.
double estimatePitch(GpuFft& fft, int windowLen, double freq, bool interpolate) {
    const int fftSize = fft.size();

    // 1. Initialize zero-padded buffer of length `fftSize`
    std::vector<std::complex<float>> signal(fftSize, {0.0f, 0.0f});

    // 2. Fill only the first `windowLen` samples with the Hann-windowed sine wave
    for (int n = 0; n < windowLen; ++n) {
        const double window = 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * n / windowLen));
        const double phase = 2.0 * std::numbers::pi * freq * n / SAMPLE_RATE;
        signal[n] = {static_cast<float>(std::sin(phase) * window), 0.0f};
    }

    // 3. Run the GPU FFT of size `fftSize`
    std::vector<std::complex<float>> spectrum(fftSize);
    fft.forward(signal.data(), spectrum.data());

    // 4. Find the strongest positive bin in [1, fftSize/2 - 1]
    int peakBin = 1;
    float peakMag = 0.0f;
    for (int k = 1; k < fftSize / 2; ++k) {
        const float mag = std::abs(spectrum[k]);
        if (mag > peakMag) {
            peakMag = mag;
            peakBin = k;
        }
    }

    const double binWidth = SAMPLE_RATE / static_cast<double>(fftSize);
    if (!interpolate) {
        return static_cast<double>(peakBin) * binWidth;
    }

    // 5. Log-magnitude parabolic interpolation
    const double alpha = std::log(static_cast<double>(std::abs(spectrum[peakBin - 1])));
    const double beta  = std::log(static_cast<double>(std::abs(spectrum[peakBin])));
    const double gamma = std::log(static_cast<double>(std::abs(spectrum[peakBin + 1])));

    const double offset = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma);
    return (static_cast<double>(peakBin) + offset) * binWidth;
}

}  // namespace

int main() {
    constexpr int WINDOW_LEN = 1024;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Window   FFT size   Latency   Worst |error|   Mean |error|   Worst freq\n";

    for (int fftSize : {1024, 2048, 4096, 8192}) {
        GpuFft fft(fftSize);

        std::ofstream csv("sweep_pad_" + std::to_string(fftSize) + ".csv");
        if (csv) {
            csv << "freq,binCents,interpCents\n";
        }

        double interpWorst = 0.0;
        double interpSum = 0.0;
        int worstFreq = 100;
        int count = 0;

        for (int f = 100; f <= 1000; ++f) {
            const double trueFreq = static_cast<double>(f);

            const double binEst = estimatePitch(fft, WINDOW_LEN, trueFreq, false);
            const double interpEst = estimatePitch(fft, WINDOW_LEN, trueFreq, true);

            const double binCents = 1200.0 * std::log2(binEst / trueFreq);
            const double interpCents = 1200.0 * std::log2(interpEst / trueFreq);

            if (std::abs(interpCents) > interpWorst) {
                interpWorst = std::abs(interpCents);
                worstFreq = f;
            }
            interpSum += std::abs(interpCents);
            ++count;

            if (csv) {
                csv << f << ',' << binCents << ',' << interpCents << '\n';
            }
        }

        const double latencyMs = (1000.0 * WINDOW_LEN) / SAMPLE_RATE;
        std::cout << std::setw(6) << WINDOW_LEN << "   "
                  << std::setw(8) << fftSize    << "   "
                  << std::setw(5) << latencyMs  << " ms   "
                  << std::setw(7) << interpWorst << " cents     "
                  << std::setw(6) << (interpSum / count) << " cents   "
                  << "(at " << worstFreq << " Hz)\n";
    }

    return EXIT_SUCCESS;
}
