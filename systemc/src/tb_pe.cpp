// Unit test + waveform of MAC PE (pipelined) and MaxPool PE
#include <systemc.h>
#include <random>
#include "pe.h"
int sc_main(int, char**) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<sc_int<8>> a, b; sc_signal<bool> valid, clear; sc_signal<sc_int<32>> acc;
    sc_signal<sc_uint<8>> i0, i1, i2, i3, mo;
    MacPE mac("mac"); mac.clk(clk); mac.a(a); mac.b(b); mac.valid(valid); mac.clear(clear); mac.acc(acc);
    MaxPoolPE mp("mp"); mp.i0(i0); mp.i1(i1); mp.i2(i2); mp.i3(i3); mp.maxout(mo);
    sc_trace_file* tf = sc_create_vcd_trace_file("pe_wave"); tf->set_time_unit(1, SC_NS);
    sc_trace(tf, clk, "clk"); sc_trace(tf, a, "a"); sc_trace(tf, b, "b"); sc_trace(tf, valid, "valid"); sc_trace(tf, clear, "clear");
    sc_trace(tf, mac.prod, "stage1_product"); sc_trace(tf, acc, "stage2_acc");
    sc_trace(tf, i0, "mp_i0"); sc_trace(tf, i1, "mp_i1"); sc_trace(tf, i2, "mp_i2"); sc_trace(tf, i3, "mp_i3"); sc_trace(tf, mo, "mp_max");
    std::mt19937 r(7); int fails = 0, windows = 200;
    valid = false; clear = false; a = 0; b = 0; sc_start(10, SC_NS);
    for (int w = 0; w < windows; w++) {
        long expect = 0;
        for (int k = 0; k < 9; k++) {                       // one 3x3 window: 9 MACs back-to-back, 1 per clock
            int av = (int)(r() % 255) - 127, bv = (int)(r() % 255) - 127; expect += av * bv;
            a = av; b = bv; valid = true; clear = (k == 0); sc_start(10, SC_NS);
        }
        valid = false; clear = false; sc_start(10, SC_NS);     // flush: result appears 2 clocks after last input
        sc_start(10, SC_NS);
        if (acc.read() != expect) { fails++; printf("MAC mismatch window %d: got %d expect %ld\n", w, acc.read().to_int(), expect); }
        unsigned v[4] = {(unsigned)(r() % 128), (unsigned)(r() % 128), (unsigned)(r() % 128), (unsigned)(r() % 128)};
        i0 = v[0]; i1 = v[1]; i2 = v[2]; i3 = v[3]; sc_start(1, SC_NS);
        unsigned m = std::max(std::max(v[0], v[1]), std::max(v[2], v[3]));
        if (mo.read() != m) { fails++; printf("MaxPool mismatch\n"); }
    }
    printf("MAC PE: %d dot-products of 9 MACs checked, MaxPool PE: %d windows checked -> %s\n", windows, windows, fails ? "FAIL" : "PASS");
    printf("MAC PE throughput: 1 MAC/clock after 2-cycle pipeline fill (waveform: pe_wave.vcd)\n");
    sc_close_vcd_trace_file(tf);
    return fails;
}
