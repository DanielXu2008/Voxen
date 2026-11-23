#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdint>
#include <numbers>

void writeHeader(std::ofstream& out, uint32_t dataSize, uint32_t sampleRate) {
    // Audio specification constants
    constexpr uint32_t SUBCHUNK1_SIZE  = 16;    // 16 for PCM
    constexpr uint16_t AUDIO_FORMAT    = 1;     // 1 = PCM
    constexpr uint16_t NUM_CHANNELS    = 1;     // 1 = Mono
    constexpr uint16_t BITS_PER_SAMPLE = 16;    // 16-bit audio

    // Derived constants
    constexpr uint16_t BYTES_PER_SAMPLE = BITS_PER_SAMPLE / 8;
    constexpr uint16_t BLOCK_ALIGN      = NUM_CHANNELS * BYTES_PER_SAMPLE;
    const uint32_t BYTE_RATE            = sampleRate * BLOCK_ALIGN;

    // Dynamic size calculated from your sample buffer
    const uint32_t chunkSize = 36 + dataSize;

    // 1. RIFF chunk descriptor (12 bytes)
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunkSize), sizeof(chunkSize));
    out.write("WAVE", 4);

    // 2. "fmt " subchunk (24 bytes)
    out.write("fmt ", 4);
    out.write(reinterpret_cast<const char*>(&SUBCHUNK1_SIZE), sizeof(SUBCHUNK1_SIZE));
    out.write(reinterpret_cast<const char*>(&AUDIO_FORMAT), sizeof(AUDIO_FORMAT));
    out.write(reinterpret_cast<const char*>(&NUM_CHANNELS), sizeof(NUM_CHANNELS));
    out.write(reinterpret_cast<const char*>(&sampleRate), sizeof(sampleRate));
    out.write(reinterpret_cast<const char*>(&BYTE_RATE), sizeof(BYTE_RATE));
    out.write(reinterpret_cast<const char*>(&BLOCK_ALIGN), sizeof(BLOCK_ALIGN));
    out.write(reinterpret_cast<const char*>(&BITS_PER_SAMPLE), sizeof(BITS_PER_SAMPLE));

    // 3. "data" subchunk header (8 bytes)
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
}

int main() {
    const double freq[8] = {261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25}; // middle C scale
    constexpr uint32_t SAMPLE_RATE = 48000;
    constexpr double PI = std::numbers::pi;
    constexpr double NOTE_SECONDS = 2.0;
    const int samplesPerNote = static_cast<int>(NOTE_SECONDS * SAMPLE_RATE);

    std::vector<int16_t> output;
    output.reserve(8 * samplesPerNote);

    double phase = 0.0;  // carried across notes so the waveform never jumps
    for (int note = 0; note < 8; note++) {
        const double step = 2 * PI * freq[note] / SAMPLE_RATE;  // phase advance per sample
        for (int i = 0; i < samplesPerNote; i++) {
            output.push_back(static_cast<int16_t>(std::lround(std::sin(phase) * 0.3 * 32767)));
            phase += step;
            if (phase >= 2 * PI) phase -= 2 * PI;  // keep phase small for precision
        }
    }

    uint32_t dataSize = static_cast<uint32_t>(output.size() * sizeof(int16_t));
    std::ofstream out("tone.wav", std::ios::binary);
    if (!out) return 1;

    writeHeader(out, dataSize, SAMPLE_RATE);
    out.write(reinterpret_cast<const char*>(output.data()), dataSize);
}
