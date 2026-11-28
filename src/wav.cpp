#include "wav.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace {

std::uint16_t readU16(const unsigned char* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t readU32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

}

WavData readWav(const std::string& path) {
    auto fail = [&](const std::string& why) { throw std::runtime_error("readWav: " + path + ": " + why); };

    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open file");
    const std::vector<unsigned char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    if (file.size() < 12 || std::memcmp(file.data(), "RIFF", 4) != 0 || std::memcmp(file.data() + 8, "WAVE", 4) != 0) {
        fail("not a RIFF/WAVE file");
    }

    int format = 0, channels = 0, bits = 0;
    std::uint32_t rate = 0;
    const unsigned char* data = nullptr;
    std::size_t dataSize = 0;

    // Walk the chunks. Editors add extra ones (LIST, bext, ...), so skip anything unknown.
    std::size_t pos = 12;
    while (pos + 8 <= file.size()) {
        const unsigned char* chunk = file.data() + pos;
        const std::uint32_t size = readU32(chunk + 4);
        const unsigned char* body = chunk + 8;
        const std::size_t available = file.size() - pos - 8;

        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (size < 16 || size > available) fail("bad fmt chunk");
            format = readU16(body);
            channels = readU16(body + 2);
            rate = readU32(body + 4);
            bits = readU16(body + 14);
            // WAVE_FORMAT_EXTENSIBLE stores the real format in the first 2 bytes of its SubFormat GUID
            if (format == 0xFFFE && size >= 26) format = readU16(body + 24);
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = body;
            dataSize = std::min<std::size_t>(size, available);  // tolerate truncated files
        }
        pos += 8 + static_cast<std::size_t>(size) + (size & 1);  // chunk bodies are padded to even length
    }

    if (format == 0) fail("no fmt chunk");
    if (data == nullptr) fail("no data chunk");
    if (channels < 1) fail("invalid channel count");

    const bool pcm16 = format == 1 && bits == 16;
    const bool pcm24 = format == 1 && bits == 24;
    const bool float32 = format == 3 && bits == 32;
    if (!pcm16 && !pcm24 && !float32) {
        fail("unsupported format " + std::to_string(format) + " with " + std::to_string(bits) + " bits");
    }

    const int bytesPerSample = bits / 8;
    const std::size_t frames = dataSize / (static_cast<std::size_t>(bytesPerSample) * channels);

    WavData wav;
    wav.sampleRate = static_cast<int>(rate);
    wav.channels = channels;
    wav.samples.resize(frames);

    for (std::size_t f = 0; f < frames; ++f) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const unsigned char* p = data + (f * channels + c) * bytesPerSample;
            if (pcm16) {
                sum += static_cast<std::int16_t>(readU16(p)) / 32768.0f;
            } else if (pcm24) {
                std::int32_t v = p[0] | (p[1] << 8) | (p[2] << 16);
                if (v & 0x800000) v -= 0x1000000;  // sign-extend 24 -> 32 bits
                sum += v / 8388608.0f;
            } else {
                float v;
                std::memcpy(&v, p, sizeof v);
                sum += v;
            }
        }
        wav.samples[f] = sum / channels;
    }
    return wav;
}
