// HPS sketch + quick check (throwaway, not in the repo).
#include "fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------- sketch starts

struct PitchEstimate {
    double hz = 0.0;          // 0 when unvoiced
    double confidence = 0.0;  // 0..1: share of in-band energy that sits on the harmonics
    bool voiced = false;
};

struct HpsConfig {
    double sampleRate = 48000.0;
    int fftSize = 4096;          // 2048-sample Hann window, zero-padded 2x
    double minHz = 80.0;         // low male voice
    double maxHz = 1000.0;       // high soprano
    int numHarmonics = 5;
    double gridCents = 5.0;      // spacing of f0 candidates
    double floorRatio = 0.01;    // magnitudes under 1% of the peak count as "nothing here"
    double minConfidence = 0.9;  // voiced/unvoiced threshold: tune on real recordings
    double minPeak = 1e-3;       // silence gate: depends on input level
};

class HpsDetector {
public:
    explicit HpsDetector(const HpsConfig& config) : config_(config) {
        if (config.maxHz * config.numHarmonics >= config.sampleRate / 2) {
            throw std::invalid_argument("HpsDetector: highest harmonic would pass Nyquist");
        }
        binHz_ = config.sampleRate / config.fftSize;
        mag_.resize(config.fftSize / 2 + 1);

        // Log-spaced candidates: equal spacing in cents means equal precision at every note.
        const double step = std::exp2(config.gridCents / 1200.0);
        for (double f = config.minHz; f <= config.maxHz; f *= step) candidates_.push_back(f);
    }

    // spectrum: full FFT output (fftSize values) of a Hann-windowed, zero-padded frame.
    PitchEstimate estimate(std::span<const std::complex<float>> spectrum) {
        if (static_cast<int>(spectrum.size()) != config_.fftSize) {
            throw std::invalid_argument("HpsDetector: spectrum size doesn't match fftSize");
        }

        double peak = 0.0;
        for (std::size_t k = 0; k < mag_.size(); ++k) {
            mag_[k] = std::abs(spectrum[k]);
            peak = std::max(peak, mag_[k]);
        }
        if (peak < config_.minPeak) return {};  // silence

        // 1. Coarse search: which f0 lines up best with the harmonic peaks?
        const double floorMag = peak * config_.floorRatio;
        double bestF0 = candidates_.front();
        double bestScore = -1e300;
        for (double f0 : candidates_) {
            const double s = score(f0, floorMag);
            if (s > bestScore) {
                bestScore = s;
                bestF0 = f0;
            }
        }

        // 2. Refine: interpolate the strongest harmonic, divide by its harmonic number.
        const double f0 = refine(bestF0);

        // 3. Voicing: is the energy concentrated on the harmonics, or spread out like noise?
        const double confidence = harmonicEnergyShare(f0);
        const bool voiced = confidence >= config_.minConfidence;
        return {voiced ? f0 : 0.0, confidence, voiced};
    }

private:
    // Magnitude at a fractional bin position (linear interpolation between neighbors).
    double magnitudeAt(double bin) const {
        const int i = static_cast<int>(bin);
        const double frac = bin - i;
        return mag_[i] * (1.0 - frac) + mag_[i + 1] * frac;
    }

    // Mean log magnitude at h*f0. A log-sum is a product in disguise, without overflow.
    // The floor stops one empty harmonic (e.g. a missing fundamental) from vetoing the candidate.
    double score(double f0, double floorMag) const {
        double sum = 0.0;
        int count = 0;
        for (int h = 1; h <= config_.numHarmonics; ++h) {
            const double bin = h * f0 / binHz_;
            if (bin >= mag_.size() - 1) break;
            sum += std::log(std::max(magnitudeAt(bin), floorMag));
            ++count;
        }
        return count > 0 ? sum / count : -1e300;
    }

    // Returns a bin that is a genuine local maximum within +/- radius of `bin`, or -1.
    int localMaxNear(double bin, int radius) const {
        const int last = static_cast<int>(mag_.size()) - 2;
        const int center = static_cast<int>(std::lround(bin));
        const int lo = std::max(1, center - radius);
        const int hi = std::min(last, center + radius);
        if (lo > hi) return -1;

        int best = lo;
        for (int k = lo + 1; k <= hi; ++k) {
            if (mag_[k] > mag_[best]) best = k;
        }
        const bool isPeak = mag_[best] >= mag_[best - 1] && mag_[best] >= mag_[best + 1];
        return isPeak ? best : -1;
    }

    // Log-magnitude parabolic interpolation around a local maximum. Returns a fractional bin.
    double interpolatePeakBin(int k) const {
        const double a = std::log(mag_[k - 1] + 1e-12);
        const double b = std::log(mag_[k] + 1e-12);
        const double g = std::log(mag_[k + 1] + 1e-12);
        const double denom = a - 2.0 * b + g;
        if (denom >= -1e-12) return k;  // flat top: no usable curvature
        return k + std::clamp(0.5 * (a - g) / denom, -0.5, 0.5);
    }

    // The coarse f0 is within a few cents, so h*f0 lands within a fraction of a bin of the
    // real harmonic. That makes the harmonic number unambiguous (unlike assigning h by
    // which search window a peak happens to fall in).
    double refine(double f0) const {
        int bestK = -1;
        int bestH = 1;
        for (int h = 1; h <= config_.numHarmonics; ++h) {
            const int k = localMaxNear(h * f0 / binHz_, 2);
            if (k >= 0 && (bestK < 0 || mag_[k] > mag_[bestK])) {
                bestK = k;
                bestH = h;
            }
        }
        if (bestK < 0) return f0;  // no clean peak anywhere: keep the coarse estimate
        return interpolatePeakBin(bestK) * binHz_ / bestH;
    }

    // Share of energy between 0.5*f0 and (H + 0.5)*f0 that lies within +/-2 bins of a harmonic.
    // A heuristic: near 1 for a clean voice, low for breath noise. Threshold must be tuned.
    double harmonicEnergyShare(double f0) const {
        const int last = static_cast<int>(mag_.size()) - 1;
        const int lo = std::max(1, static_cast<int>(0.5 * f0 / binHz_));
        const int hi = std::min(last, static_cast<int>((config_.numHarmonics + 0.5) * f0 / binHz_));

        double band = 0.0;
        for (int k = lo; k <= hi; ++k) band += mag_[k] * mag_[k];

        double onHarmonics = 0.0;
        for (int h = 1; h <= config_.numHarmonics; ++h) {
            const int c = static_cast<int>(std::lround(h * f0 / binHz_));
            for (int k = std::max(lo, c - 2); k <= std::min(hi, c + 2); ++k) onHarmonics += mag_[k] * mag_[k];
        }
        return band > 0.0 ? onHarmonics / band : 0.0;
    }

    HpsConfig config_;
    double binHz_ = 0.0;
    std::vector<double> candidates_;  // precomputed once
    std::vector<double> mag_;         // reused every frame: no allocation per call
};

// ---------------------------------------------------------------- sketch ends

std::vector<std::complex<float>> frame(double f0, const std::vector<double>& amps, double noise, unsigned seed) {
    std::vector<std::complex<float>> x(4096, {0, 0});
    unsigned s = seed;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) / double(1 << 24) - 0.5; };
    for (int n = 0; n < 2048; ++n) {
        double v = noise * rnd();
        for (std::size_t h = 0; h < amps.size(); ++h)
            v += 0.4 * amps[h] * std::sin(2 * std::numbers::pi * (h + 1) * f0 * n / 48000.0);
        const double w = 0.5 * (1 - std::cos(2 * std::numbers::pi * n / 2048));
        x[n] = {float(v * w), 0};
    }
    return x;
}

int main() {
    GpuFft fft(4096);
    HpsDetector hps(HpsConfig{});
    std::vector<std::complex<float>> spec(4096);

    struct Case { const char* name; std::vector<double> amps; double noise; };
    const Case cases[] = {
        {"weak fundamental", {0.3, 1.0, 0.6, 0.3}, 0.0},
        {"missing fundamental", {0.0, 1.0, 0.6, 0.3}, 0.0},
        {"missing fund + noise", {0.0, 1.0, 0.6, 0.3}, 0.2},
    };
    for (const auto& c : cases) {
        int fails = 0, unvoiced = 0, count = 0;
        double worst = 0, sum = 0, minConf = 1;
        for (int f = 80; f <= 1000; ++f) {
            const auto x = frame(f, c.amps, c.noise, f);
            fft.forward(x.data(), spec.data());
            const auto p = hps.estimate(spec);
            minConf = std::min(minConf, p.confidence);
            if (!p.voiced) { ++unvoiced; continue; }
            const double cents = 1200 * std::log2(p.hz / f);
            if (std::abs(cents) > 50) { ++fails; continue; }
            worst = std::max(worst, std::abs(cents)); sum += std::abs(cents); ++count;
        }
        std::printf("%-22s fails %d, unvoiced %d, worst %.2f c, mean %.3f c, min conf %.2f\n",
                    c.name, fails, unvoiced, worst, count ? sum / count : 0.0, minConf);
    }
    // Pure noise should come out unvoiced
    int noiseVoiced = 0;
    double maxConf = 0;
    for (int i = 0; i < 200; ++i) {
        const auto x = frame(100, {}, 0.5, 1000 + i);
        fft.forward(x.data(), spec.data());
        const auto p = hps.estimate(spec);
        noiseVoiced += p.voiced;
        maxConf = std::max(maxConf, p.confidence);
    }
    std::printf("pure noise: %d / 200 frames called voiced, max confidence %.2f\n", noiseVoiced, maxConf);
}
