// CNN accelerator: FSM (Idle -> Load -> layers -> Argmax -> Done), timing from timing.h, bit-exact maths from model.h
#pragma once
#include <systemc.h>
#include <array>
#include "model.h"
#include "timing.h"
#include "auth.h"

SC_MODULE(CnnAccel) {
    sc_in<bool> clk, start, authorized, privacy;
    sc_in<sc_uint<32>> img_addr;
    sc_in<sc_uint<4>> noise_in;
    sc_out<bool> done, busy;
    sc_out<sc_uint<4>> result, fsm_state;   // 0 IDLE 1 LOAD 2 CONV1 3 POOL1 4 CONV2 5 POOL2 6 DENSE1 7 DENSE2 8 ARGMAX 9 DONE

    const Model* model = nullptr;
    std::vector<uint8_t>* mem = nullptr;
    TimingCfg cfg;
    bool keyed_mask = false;                // improvement: keyed LFSR mask the legitimate CPU can undo
    Lfsr4 klfsr;
    std::vector<std::array<int32_t, 10>> logits_log;
    std::vector<long> cycles_log;
    std::vector<Sched> first_sched;
    long refused = 0, n_inf = 0; bool prev_start = false;

    void run() {
        done.write(false); busy.write(false); fsm_state.write(0); result.write(0);
        while (true) {
            wait();
            if (!start.read()) { prev_start = false; continue; }
            if (!authorized.read()) { if (!prev_start) refused++; prev_start = true; continue; }          // hardware refuses un-authorised start
            uint32_t a = img_addr.read();
            if (!mem || a + 784 > mem->size()) { refused++; continue; }
            busy.write(true); fsm_state.write(1);
            auto sched = schedule(cfg);
            if (first_sched.empty()) first_sched = sched;
            long cur = 0;
            wait(cfg.load_cycles); cur += cfg.load_cycles;
            int32_t lg[10];
            model->infer(mem->data() + a, lg);                            // functional result of all layers
            for (size_t i = 0; i < sched.size(); i++) {
                fsm_state.write((unsigned)(2 + i));
                long target = cfg.load_cycles + sched[i].end;
                if (target > cur) { wait(target - cur); cur = target; }
            }
            unsigned pred = argmax10(lg);
            if (privacy.read()) {
                unsigned m = keyed_mask ? klfsr.step() : noise_in.read().to_uint();
                pred ^= m;                                                   // final_output = class XOR lfsr_noise
            }
            wait(2); cur += 2;
            result.write(pred); fsm_state.write(9); done.write(true);
            std::array<int32_t, 10> arr; for (int i = 0; i < 10; i++) arr[i] = lg[i];
            logits_log.push_back(arr); cycles_log.push_back(cur); n_inf++;
            wait();
            done.write(false); busy.write(false); fsm_state.write(0);
        }
    }
    SC_CTOR(CnnAccel) {
        SC_CTHREAD(run, clk.pos());
    }
};
