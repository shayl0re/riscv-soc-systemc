// Security / privacy blocks: authentication FSM (paper Fig. 6) and LFSR noise source (Fig. 7)
#pragma once
#include <systemc.h>

// Signature function (paper: "simple XOR and bit-shift logic"): sig = key XOR rotate_right(challenge,1)
inline unsigned sig_fn(unsigned key, unsigned chal) { return (key ^ (((chal >> 1) | ((chal & 1) << 3)) & 0xF)) & 0xF; }

// 4-bit Fibonacci LFSR, polynomial x^4 + x^3 + 1 (period 15)
struct Lfsr4 {
    unsigned s;
    explicit Lfsr4(unsigned seed = 9) : s(seed ? seed & 0xF : 1) {}
    unsigned step() { unsigned b = ((s >> 3) ^ (s >> 2)) & 1; s = ((s << 1) | b) & 0xF; return s; }
};

SC_MODULE(AuthFSM) {
    sc_in<bool> clk, req;
    sc_in<sc_uint<4>> key, chal, sig;
    sc_out<sc_uint<2>> state;           // 0 IDLE, 1 VERIFY, 2 UNLOCKED, 3 ERROR
    sc_out<bool> unlocked, error;
    unsigned secret_key = 0xA;
    bool hardened = false;              // improvement: reject non-fresh (replayed) challenges
    int last_chal = -1;
    long n_ok = 0, n_fail = 0;
    unsigned st = 0;
    void proc() {
        switch (st) {
            case 0: if (req.read()) st = 1; break;
            case 1: {
                unsigned k = key.read(), c = chal.read(), s = sig.read();
                bool ok = (k == secret_key) && (s == sig_fn(k, c));
                if (ok && hardened && last_chal >= 0) { int d = (int)((c - last_chal) & 15); if (d < 1 || d > 8) ok = false; }
                if (ok) { last_chal = c; n_ok++; st = 2; } else { n_fail++; st = 3; }
                break; }
            default: if (!req.read()) st = 0; break;   // graceful: no crash, just back to IDLE
        }
        state.write(st); unlocked.write(st == 2); error.write(st == 3);
    }
    SC_CTOR(AuthFSM) { SC_METHOD(proc); sensitive << clk.pos(); dont_initialize(); }
};

// Free-running LFSR (the paper's noise source)
SC_MODULE(LfsrNoise) {
    sc_in<bool> clk;
    sc_out<sc_uint<4>> noise;
    Lfsr4 l;
    void proc() { noise.write(l.step()); }
    SC_CTOR(LfsrNoise) : l(9) { SC_METHOD(proc); sensitive << clk.pos(); dont_initialize(); }
};
