// Cycle-count (latency) model of the accelerator: PE parallelism, memory limit, optional inter-layer pipelining
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

enum Kind { CONV, POOL, DENSE, ARGMAX };
struct LayerDesc { const char* name; Kind kind; long macs; long outputs; int out_h; int rows_needed; };

// MNIST network of the paper. rows_needed: rows of previous layer needed before this layer can start (0 = all)
inline const std::vector<LayerDesc>& mnist_layers() {
    static const std::vector<LayerDesc> L = {
        {"Conv 3x3x32",   CONV,  194688, 21632, 26, 3},
        {"MaxPool 2x2x32", POOL,      0,  5408, 13, 2},
        {"Conv 3x3x64",   CONV, 2230272,  7744, 11, 3},
        {"MaxPool 2x2x64", POOL,      0,  1600,  5, 2},
        {"Dense 1600-128", DENSE, 204800,  128,  1, 0},
        {"Dense 128-10",  DENSE,   1280,    10,  1, 0},
        {"Argmax",        ARGMAX,     0,    10,  1, 0}};
    return L;
}

struct TimingCfg {
    int n_pe = 128;          // MAC PEs: 1, 9 or 128
    bool pipelined = false;  // inter-layer pipelining
    bool shared_mac = true;  // one shared MAC array (paper Fig.1). false = ideal: each layer has its own array
    int mem_macs = 63;       // max MACs/cycle the BRAM bandwidth can feed (calibrated to paper Table I 9PE->128PE ratio)
    int overhead = 16;       // FSM / buffer hand-off cycles per layer
    int load_cycles = 196;   // 784 bytes at 4 bytes/cycle
    double clk_mhz = 15.0;   // implied by paper Table I (not stated in the paper)
};
struct Sched { long start, end, T; };

inline long ceil_div(long a, long b) { return (a + b - 1) / b; }
inline long layer_cycles(const LayerDesc& l, const TimingCfg& c) {
    long peff = std::min(c.n_pe, c.mem_macs), t;
    switch (l.kind) {
        case CONV: case DENSE: t = ceil_div(l.macs, peff) + 2; break;   // +2 = MAC pipeline fill (mult, acc)
        case POOL: t = ceil_div(l.outputs, std::max(1, c.n_pe / 9)) + 1; break;
        default: t = 11;
    }
    return t + c.overhead;
}
inline std::vector<Sched> schedule(const TimingCfg& c) {
    const auto& L = mnist_layers();
    std::vector<Sched> s;
    long mac_free = 0;
    for (size_t i = 0; i < L.size(); i++) {
        long T = layer_cycles(L[i], c), start = 0, end;
        if (i > 0) {
            long pe = s[i - 1].end, ps = s[i - 1].start, pT = s[i - 1].T;
            if (!c.pipelined) start = pe;
            else {
                int r = L[i].rows_needed;
                long su = r == 0 ? pT : ceil_div(pT * r, L[i - 1].out_h);
                start = ps + su;
            }
        }
        bool is_mac = L[i].kind == CONV || L[i].kind == DENSE;
        if (c.pipelined && c.shared_mac && is_mac) start = std::max(start, mac_free);
        end = start + T;
        if (c.pipelined && i > 0) end = std::max(end, s[i - 1].end + ceil_div(T, L[i].out_h));  // drain of last row
        if (is_mac) mac_free = end;
        s.push_back({start, end, T});
    }
    return s;
}
inline long total_cycles(const TimingCfg& c) { return c.load_cycles + schedule(c).back().end + 2; }
