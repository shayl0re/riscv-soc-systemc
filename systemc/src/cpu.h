// RV32IM instruction-set simulator as an SC_THREAD, cycle-approximate (1 cycle/instr, +2 taken branch, +2 mul/div)
// plus the paper's custom CNN instruction: opcode 1111011, rd=result, rs1=image pointer,
// funct3 = {privacy, cnn_variant[1:0]}, imm12 = {key[3:0], challenge[3:0], signature[3:0]}
#pragma once
#include <systemc.h>
#include <vector>
#include "auth.h"

enum { OP_CNN = 0x7B };
constexpr uint32_t RES_DENIED = 0xFFFFFFFFu, RES_BYPASS = 0xFFFFFFFDu, RES_UNSUP = 0xFFFFFFFEu;

struct CnnTrace { int idx; bool ok; uint32_t result; long cycles; };

SC_MODULE(RiscvCpu) {
    sc_in<bool> clk;
    sc_out<bool> auth_req, cnn_start, cnn_priv;
    sc_out<sc_uint<4>> key, chal, sig;
    sc_out<sc_uint<32>> img_addr, pc_out;
    sc_in<bool> auth_unlocked, auth_error, cnn_done;
    sc_in<sc_uint<4>> cnn_result;

    std::vector<uint32_t> imem;
    std::vector<uint8_t>* dmem = nullptr;
    uint32_t x[32] = {0}, pc = 0;
    bool bypass = false;                  // attacker model: raw start strobe without authentication
    long cyc = 0, instret = 0, cnn_calls = 0, cnn_denied = 0;
    std::vector<CnnTrace> traces;
    sc_time period;

    void tick(int n = 1) { for (int i = 0; i < n; i++) { wait(clk.posedge_event()); cyc++; } }
    static int32_t sext(uint32_t v, int bits) { return (int32_t)(v << (32 - bits)) >> (32 - bits); }
    uint32_t ld(uint32_t a, int n, bool sgn) {
        if (a + n > dmem->size()) return 0;
        uint32_t v = 0; for (int i = n - 1; i >= 0; i--) v = (v << 8) | (*dmem)[a + i];
        return sgn ? (uint32_t)sext(v, 8 * n) : v;
    }
    void st(uint32_t a, uint32_t v, int n) { if (a + n > dmem->size()) return; for (int i = 0; i < n; i++) (*dmem)[a + i] = (v >> (8 * i)) & 0xFF; }

    void custom(uint32_t ins) {
        uint32_t imm = (ins >> 20) & 0xFFF, f3 = (ins >> 12) & 7, rd = (ins >> 7) & 31, rs1 = (ins >> 15) & 31;
        long t0 = cyc; cnn_calls++;
        uint32_t res = 0; bool ok = false;
        img_addr.write(x[rs1]); key.write((imm >> 8) & 0xF); chal.write((imm >> 4) & 0xF); sig.write(imm & 0xF);
        cnn_priv.write((f3 >> 2) & 1);
        bool variant_ok = (f3 & 3) == 0;                                   // only MNIST implemented
        if (!bypass) {
            auth_req.write(true); tick(1);
            int g = 0; while (!(auth_unlocked.read() || auth_error.read()) && g++ < 8) tick(1);
            ok = auth_unlocked.read();
        } else ok = true;                                                  // attacker skips the auth step
        if (!ok) { res = RES_DENIED; cnn_denied++; }
        else if (!variant_ok) { res = RES_UNSUP; ok = false; }
        else {
            cnn_start.write(true); tick(2); cnn_start.write(false);
            if (!cnn_done.read()) wait(period * (bypass ? 50 : 20000000), cnn_done.posedge_event());
            if (cnn_done.read()) { res = cnn_result.read().to_uint(); cyc = (long)(sc_time_stamp() / period + 0.5); }
            else { res = RES_BYPASS; ok = false; cnn_denied++; }
        }
        auth_req.write(false); tick(1);
        if (rd) x[rd] = res;
        traces.push_back({(int)traces.size(), ok, res, cyc - t0});
    }

    void run() {
        auth_req.write(false); cnn_start.write(false); cnn_priv.write(false);
        tick(1);
        while (true) {
            pc_out.write(pc);
            uint32_t idx = pc >> 2;
            if (idx >= imem.size()) break;
            uint32_t ins = imem[idx], op = ins & 0x7F, rd = (ins >> 7) & 31, f3 = (ins >> 12) & 7, rs1 = (ins >> 15) & 31, rs2 = (ins >> 20) & 31, f7 = ins >> 25;
            int32_t immI = sext(ins >> 20, 12), immS = sext(((ins >> 25) << 5) | ((ins >> 7) & 31), 12);
            int32_t immB = sext(((ins >> 31) << 12) | (((ins >> 7) & 1) << 11) | (((ins >> 25) & 63) << 5) | (((ins >> 8) & 15) << 1), 13);
            int32_t immJ = sext(((ins >> 31) << 20) | (((ins >> 12) & 255) << 12) | (((ins >> 20) & 1) << 11) | (((ins >> 21) & 1023) << 1), 21);
            uint32_t npc = pc + 4, w = 0; int extra = 0; bool wr = true;
            uint32_t a = x[rs1], b = x[rs2];
            switch (op) {
                case 0x37: w = ins & 0xFFFFF000; break;
                case 0x17: w = pc + (ins & 0xFFFFF000); break;
                case 0x6F: w = pc + 4; npc = pc + immJ; extra = 2; break;
                case 0x67: w = pc + 4; npc = (a + immI) & ~1u; extra = 2; break;
                case 0x63: { wr = false; bool t = false;
                    switch (f3) { case 0: t = a == b; break; case 1: t = a != b; break; case 4: t = (int32_t)a < (int32_t)b; break;
                                  case 5: t = (int32_t)a >= (int32_t)b; break; case 6: t = a < b; break; case 7: t = a >= b; break; }
                    if (t) { npc = pc + immB; extra = 2; } break; }
                case 0x03: switch (f3) { case 0: w = ld(a + immI, 1, true); break; case 1: w = ld(a + immI, 2, true); break;
                                         case 2: w = ld(a + immI, 4, false); break; case 4: w = ld(a + immI, 1, false); break;
                                         case 5: w = ld(a + immI, 2, false); break; } break;
                case 0x23: wr = false; st(a + immS, b, f3 == 0 ? 1 : f3 == 1 ? 2 : 4); break;
                case 0x13: switch (f3) { case 0: w = a + immI; break; case 2: w = (int32_t)a < immI; break; case 3: w = a < (uint32_t)immI; break;
                                         case 4: w = a ^ immI; break; case 6: w = a | immI; break; case 7: w = a & immI; break;
                                         case 1: w = a << (immI & 31); break; case 5: w = (f7 & 0x20) ? (uint32_t)((int32_t)a >> (immI & 31)) : a >> (immI & 31); break; } break;
                case 0x33:
                    if (f7 == 1) { extra = 2; switch (f3) { case 0: w = a * b; break; case 4: w = b ? (int32_t)a / (int32_t)b : -1; break; case 5: w = b ? a / b : ~0u; break;
                                                              case 6: w = b ? (int32_t)a % (int32_t)b : a; break; case 7: w = b ? a % b : a; break; default: w = 0; } }
                    else switch (f3) { case 0: w = (f7 & 0x20) ? a - b : a + b; break; case 1: w = a << (b & 31); break; case 2: w = (int32_t)a < (int32_t)b; break;
                                       case 3: w = a < b; break; case 4: w = a ^ b; break; case 5: w = (f7 & 0x20) ? (uint32_t)((int32_t)a >> (b & 31)) : a >> (b & 31); break;
                                       case 6: w = a | b; break; case 7: w = a & b; break; } break;
                case OP_CNN: wr = false; custom(ins); break;
                case 0x73: instret++; pc_out.write(pc); sc_stop(); return;   // ECALL = halt
                default: wr = false; break;
            }
            if (wr && rd) x[rd] = w;
            x[0] = 0; pc = npc; instret++;
            if (op != OP_CNN) tick(1 + extra);
        }
        sc_stop();
    }
    SC_CTOR(RiscvCpu) : period(SC_ZERO_TIME) { SC_THREAD(run); }
};

// ---- tiny assembler helpers ----
inline uint32_t enc_i(uint32_t op, uint32_t rd, uint32_t f3, uint32_t rs1, int32_t imm) { return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op; }
inline uint32_t enc_s(uint32_t f3, uint32_t rs1, uint32_t rs2, int32_t imm) { return (((imm >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 31) << 7) | 0x23; }
inline uint32_t enc_u(uint32_t op, uint32_t rd, uint32_t imm20) { return (imm20 << 12) | (rd << 7) | op; }
inline void emit_li(std::vector<uint32_t>& p, uint32_t rd, uint32_t v) {
    int32_t lo = (int32_t)(v << 20) >> 20;
    p.push_back(enc_u(0x37, rd, (v - lo) >> 12)); p.push_back(enc_i(0x13, rd, 0, rd, lo));
}
inline uint32_t enc_cnn(uint32_t rd, uint32_t rs1, uint32_t f3, uint32_t key, uint32_t chal, uint32_t sg) {
    uint32_t imm = ((key & 15) << 8) | ((chal & 15) << 4) | (sg & 15);
    return (imm << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | OP_CNN;
}
