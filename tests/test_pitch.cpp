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

// Returns the estimated frequency in Hz.
double estimatePitch(GpuFft& fft, int N, double freq, bool interpolate) {
    // 1. Generate Hann-windowed sine wave at `freq`
    std::vector<std::complex<float>> signal(N);
    for (int n = 0; n < N; ++n) {
        const double window = 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * n / N));
        const double phase = 2.0 * std::numbers::pi * freq * n / SAMPLE_RATE;
        signal[n] = {static_cast<float>(std::sin(phase) * window), 0.0f};
    }

    // 2. Run the GPU FFT
    std::vector<std::complex<float>> spectrum(N);
    fft.forward(signal.data(), spectrum.data());

    // 3. Find the strongest positive bin in [1, N/2 - 1]
    int peakBin = 1;
    float peakMag = 0.0f;
    for (int k = 1; k < N / 2; ++k) {
        const float mag = std::abs(spectrum[k]);
        if (mag > peakMag) {
            peakMag = mag;
            peakBin = k;
        }
    }

    const double binWidth = SAMPLE_RATE / static_cast<double>(N);
    if (!interpolate) {
        return static_cast<double>(peakBin) * binWidth;
    }

    // 4. Log-magnitude parabolic interpolation
    const double alpha = std::log(static_cast<double>(std::abs(spectrum[peakBin - 1])));
    const double beta  = std::log(static_cast<double>(std::abs(spectrum[peakBin])));
    const double gamma = std::log(static_cast<double>(std::abs(spectrum[peakBin + 1])));

    const double offset = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma);
    return (static_cast<double>(peakBin) + offset) * binWidth;
}

}  // namespace

int main() {
    for (int N : {1024, 2048, 4096}) {
        GpuFft fft(N);

        std::ofstream csv("sweep_" + std::to_string(N) + ".csv");
        if (!csv) {
            std::cerr << "Warning: failed to open sweep_" << N << ".csv\n";
        } else {
            csv << "freq,binCents,interpCents\n";
        }

        double binWorst = 0.0;
        double binSum = 0.0;
        double interpWorst = 0.0;
        double interpSum = 0.0;
        int count = 0;

        for (int f = 100; f <= 1000; ++f) {
            const double trueFreq = static_cast<double>(f);

            const double binEst = estimatePitch(fft, N, trueFreq, false);
            const double interpEst = estimatePitch(fft, N, trueFreq, true);

            const double binCents = 1200.0 * std::log2(binEst / trueFreq);
            const double interpCents = 1200.0 * std::log2(interpEst / trueFreq);

            binWorst = std::max(binWorst, std::abs(binCents));
            binSum += std::abs(binCents);

            interpWorst = std::max(interpWorst, std::abs(interpCents));
            interpSum += std::abs(interpCents);
            ++count;

            if (csv) {
                csv << f << ',' << binCents << ',' << interpCents << '\n';
            }
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "=== N = " << N << " (" << (1000.0 * N / SAMPLE_RATE) << " ms, "
                  << (SAMPLE_RATE / N) << " Hz/bin) ===\n";
        std::cout << "method          worst |error|   mean |error|\n";
        std::cout << "bin only        " << std::setw(6) << binWorst    << " cents     " << std::setw(6) << (binSum / count)    << " cents\n";
        std::cout << "interpolated    " << std::setw(6) << interpWorst << " cents     " << std::setw(6) << (interpSum / count) << " cents\n\n";
    }

    return EXIT_SUCCESS;
}
