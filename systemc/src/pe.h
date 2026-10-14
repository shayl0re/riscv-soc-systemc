// Processing elements of the paper (Fig. 2): pipelined MAC PE and combinational 2x2 max-pool PE
#pragma once
#include <systemc.h>

// MAC PE: stage 1 = 8x8 multiply register, stage 2 = accumulate register. One MAC per clock after 2-cycle fill.
SC_MODULE(MacPE) {
    sc_in<bool> clk;
    sc_in<sc_int<8>> a, b;
    sc_in<bool> valid, clear;           // clear = first element of a new dot product
    sc_out<sc_int<32>> acc;
    sc_signal<sc_int<16>> prod;         // pipeline register 1
    sc_signal<bool> v1, c1;             // control pipeline
    sc_int<32> accr;
    void proc() {
        prod.write((sc_int<16>)(a.read() * b.read()));
        v1.write(valid.read());
        c1.write(clear.read());
        if (v1.read()) {
            accr = c1.read() ? (sc_int<32>)prod.read() : (sc_int<32>)(accr + prod.read());
            acc.write(accr);
        }
    }
    SC_CTOR(MacPE) : accr(0) { SC_METHOD(proc); sensitive << clk.pos(); dont_initialize(); }
};

// MP PE: 4 inputs -> max (pure combinational, 3 comparators + muxes)
SC_MODULE(MaxPoolPE) {
    sc_in<sc_uint<8>> i0, i1, i2, i3;
    sc_out<sc_uint<8>> maxout;
    void proc() {
        sc_uint<8> m01 = i0.read() > i1.read() ? i0.read() : i1.read();
        sc_uint<8> m23 = i2.read() > i3.read() ? i2.read() : i3.read();
        maxout.write(m01 > m23 ? m01 : m23);
    }
    SC_CTOR(MaxPoolPE) { SC_METHOD(proc); sensitive << i0 << i1 << i2 << i3; }
};
