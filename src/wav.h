#pragma once

#include <string>
#include <vector>

struct WavData {
    std::vector<float> samples;  // mono, range -1..1 (multi-channel files are averaged)
    int sampleRate = 0;
    int channels = 0;            // channel count in the file
};

// Reads 16-bit PCM, 24-bit PCM, or 32-bit float WAV. Throws std::runtime_error on failure.
WavData readWav(const std::string& path);
