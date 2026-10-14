// Bit-exact int8 CNN (same maths as python/quantize_export.py) + layer descriptors for the timing model
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "model_params.h"

inline int32_t requant(int64_t acc, int32_t M) {
    int64_t v = (acc * (int64_t)M + (int64_t(1) << (REQ_SHIFT - 1))) >> REQ_SHIFT;
    return (int32_t)std::min<int64_t>(127, std::max<int64_t>(0, v));
}

struct Model {
    std::vector<int8_t> w1, w2, w3, w4;
    std::vector<int32_t> b1, b2, b3, b4;

    bool load(const std::string& dir) {
        FILE* f = fopen((dir + "/weights.bin").c_str(), "rb");
        if (!f) return false;
        auto rd8 = [&](std::vector<int8_t>& v, size_t n) { v.resize(n); return fread(v.data(), 1, n, f) == n; };
        auto rd32 = [&](std::vector<int32_t>& v, size_t n) { v.resize(n); return fread(v.data(), 4, n, f) == n; };
        bool ok = rd8(w1, 32 * 9) && rd8(w2, 64 * 32 * 9) && rd8(w3, 128 * 1600) && rd8(w4, 10 * 128) &&
                  rd32(b1, 32) && rd32(b2, 64) && rd32(b3, 128) && rd32(b4, 10);
        fclose(f);
        return ok;
    }

    static void conv3x3(const std::vector<int32_t>& in, int C, int H, int W, const std::vector<int8_t>& w,
                        const std::vector<int32_t>& b, int F, int32_t M, std::vector<int32_t>& out) {
        int Ho = H - 2, Wo = W - 2;
        out.assign((size_t)F * Ho * Wo, 0);
        for (int f = 0; f < F; f++)
            for (int y = 0; y < Ho; y++)
                for (int x = 0; x < Wo; x++) {
                    int32_t acc = b[f];
                    for (int c = 0; c < C; c++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                                acc += in[((size_t)c * H + y + ky) * W + x + kx] * (int32_t)w[(((size_t)f * C + c) * 3 + ky) * 3 + kx];
                    out[((size_t)f * Ho + y) * Wo + x] = requant(acc, M);
                }
    }
    static void pool2(const std::vector<int32_t>& in, int C, int H, int W, std::vector<int32_t>& out) {
        int Ho = H / 2, Wo = W / 2;
        out.assign((size_t)C * Ho * Wo, 0);
        for (int c = 0; c < C; c++)
            for (int y = 0; y < Ho; y++)
                for (int x = 0; x < Wo; x++) {
                    auto at = [&](int yy, int xx) { return in[((size_t)c * H + yy) * W + xx]; };
                    out[((size_t)c * Ho + y) * Wo + x] =
                        std::max(std::max(at(2 * y, 2 * x), at(2 * y, 2 * x + 1)), std::max(at(2 * y + 1, 2 * x), at(2 * y + 1, 2 * x + 1)));
                }
    }
    // img: 784 uint8 pixels. logits: 10 x int32
    void infer(const uint8_t* img, int32_t logits[10]) const {
        std::vector<int32_t> a(784), t, u;
        for (int i = 0; i < 784; i++) a[i] = (254 * (int)img[i] + 255) / 510;  // round(pix/255*127)
        conv3x3(a, 1, 28, 28, w1, b1, 32, REQ_M1, t);   // 32x26x26
        pool2(t, 32, 26, 26, u);                         // 32x13x13
        conv3x3(u, 32, 13, 13, w2, b2, 64, REQ_M2, t);  // 64x11x11
        pool2(t, 64, 11, 11, u);                         // 64x5x5 -> flatten 1600
        std::vector<int32_t> d1(128);
        for (int o = 0; o < 128; o++) {
            int64_t acc = b3[o];
            for (int i = 0; i < 1600; i++) acc += (int64_t)u[i] * w3[(size_t)o * 1600 + i];
            d1[o] = requant(acc, REQ_M3);
        }
        for (int o = 0; o < 10; o++) {
            int64_t acc = b4[o];
            for (int i = 0; i < 128; i++) acc += (int64_t)d1[i] * w4[o * 128 + i];
            logits[o] = (int32_t)acc;
        }
    }
};
inline int argmax10(const int32_t* l) { int b = 0; for (int i = 1; i < 10; i++) if (l[i] > l[b]) b = i; return b; }
