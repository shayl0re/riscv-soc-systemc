// Top level: RISC-V CPU + Auth FSM + CNN accelerator + LFSR noise, with attack / privacy scenarios.
#include <systemc.h>
#include <cstring>
#include <cstdio>
#include <string>
#include <random>
#include <set>
#include "cpu.h"
#include "accel.h"

static const uint32_t IMG_BASE = 0x1000, RES_BASE = 0x30000, MEM_SIZE = 0x40000;
struct Call { int img; uint32_t f3, key, chal, sg; };

static std::vector<uint8_t> readfile(const std::string& p) {
    std::vector<uint8_t> v; FILE* f = fopen(p.c_str(), "rb"); if (!f) return v;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); v.resize(n); if (fread(v.data(), 1, n, f) != (size_t)n) v.clear(); fclose(f); return v;
}

static void print_table(double mhz) {
    printf("pe,mode,layer,cycles_exposed,ms\n");
    for (int pe : {1, 9, 128})
        for (int m = 0; m < 3; m++) {
            TimingCfg c; c.n_pe = pe; c.clk_mhz = mhz; c.pipelined = m > 0; c.shared_mac = (m != 2);
            const char* nm[] = {"sequential", "pipelined_shared", "pipelined_ideal"};
            auto s = schedule(c); long prev = c.load_cycles;
            for (size_t i = 0; i < s.size(); i++) { long e = s[i].end + c.load_cycles; printf("%d,%s,%s,%ld,%.4f\n", pe, nm[m], mnist_layers()[i].name, e - prev, (e - prev) / (mhz * 1e3)); prev = e; }
            printf("%d,%s,TOTAL,%ld,%.4f\n", pe, nm[m], total_cycles(c), total_cycles(c) / (mhz * 1e3));
        }
}

int sc_main(int argc, char** argv) {
    int pe = 128, n = 5, start0 = 0; bool pipe = false, ideal = false, hardened = false, vcd = false, layers = false, quiet = false, table = false;
    std::string scen = "normal", data = "data", vcdname = "soc", csv, imgfile; double mhz = 15.0; unsigned seed = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--pe") pe = atoi(nx().c_str()); else if (a == "--n") n = atoi(nx().c_str()); else if (a == "--start") start0 = atoi(nx().c_str());
        else if (a == "--pipe") pipe = true; else if (a == "--ideal-pipe") { pipe = true; ideal = true; } else if (a == "--hardened") hardened = true;
        else if (a == "--scenario") scen = nx(); else if (a == "--data") data = nx(); else if (a == "--vcd") { vcd = true; vcdname = nx(); }
        else if (a == "--layers") layers = true; else if (a == "--quiet") quiet = true; else if (a == "--table") table = true;
        else if (a == "--csv") csv = nx(); else if (a == "--image") imgfile = nx(); else if (a == "--mhz") mhz = atof(nx().c_str()); else if (a == "--seed") seed = atoi(nx().c_str());
        else { printf("unknown option %s\nusage: soc_sim [--scenario normal|wrongkey|wrongsig|tamper|replay|bypass|brute|smart|privacy] [--pe 1|9|128] [--pipe|--ideal-pipe]\n"
                      "       [--n N] [--start I] [--hardened] [--vcd name] [--layers] [--table] [--csv file] [--mhz F] [--quiet]\n", a.c_str()); return 1; }
    }
    if (table) { print_table(mhz); return 0; }

    Model model;
    if (!model.load(data)) { printf("cannot load %s/weights.bin (run python/quantize_export.py first)\n", data.c_str()); return 1; }
    auto imgs = readfile(data + "/test_images.bin"), labs = readfile(data + "/test_labels.bin"), gold = readfile(data + "/golden_logits.bin");
    if (imgs.size() != (size_t)NUM_TEST * 784) { printf("bad test data\n"); return 1; }
    const int32_t* glog = (const int32_t*)gold.data();
    bool custom = false;
    if (!imgfile.empty()) {   // live demo: replace test image 0 with a user-drawn 28x28 (784 raw bytes)
        auto u = readfile(imgfile); if (u.size() != 784) { printf("--image needs 784 raw bytes\n"); return 1; }
        memcpy(imgs.data(), u.data(), 784); labs[0] = 255; custom = true; start0 = 0; n = 1;
    }
    std::vector<uint8_t> mem(MEM_SIZE, 0);
    memcpy(mem.data() + IMG_BASE, imgs.data(), imgs.size());

    // ---------- build the call list (what the assembly program will issue) ----------
    const uint32_t KEY = 0xA; std::vector<Call> calls; std::mt19937 rng(seed);
    bool priv = scen == "privacy"; uint32_t f3 = priv ? 4 : 0;
    auto legit = [&](int i) { uint32_t c = (i + 1) & 15; return Call{(start0 + i) % NUM_TEST, f3, KEY, c, sig_fn(KEY, c)}; };
    if (scen == "normal" || priv || scen == "bypass") for (int i = 0; i < n; i++) calls.push_back(legit(i));
    else if (scen == "wrongkey") for (int i = 0; i < n; i++) { Call c = legit(i); c.key = KEY ^ 0x3; c.sg = sig_fn(c.key, c.chal); calls.push_back(c); }
    else if (scen == "wrongsig") for (int i = 0; i < n; i++) { Call c = legit(i); c.sg ^= 1; calls.push_back(c); }
    else if (scen == "tamper") for (int i = 0; i < n; i++) { Call c = legit(i); uint32_t imm = (c.key << 8) | (c.chal << 4) | c.sg; imm ^= 1u << (rng() % 12);
                                                              c.key = imm >> 8; c.chal = (imm >> 4) & 15; c.sg = imm & 15; calls.push_back(c); }
    else if (scen == "replay") { Call c = legit(2); for (int i = 0; i < 8; i++) { Call d = c; d.img = (start0 + i) % NUM_TEST; calls.push_back(d); } }
    else if (scen == "brute") for (uint32_t imm = 0; imm < 4096; imm++) calls.push_back({start0, 0, imm >> 8, (imm >> 4) & 15, imm & 15});
    else if (scen == "smart") for (uint32_t k = 0; k < 16; k++) calls.push_back({start0, 0, k, 0, sig_fn(k, 0)});
    else { printf("unknown scenario %s\n", scen.c_str()); return 1; }

    std::vector<uint32_t> prog; emit_li(prog, 6, RES_BASE);
    for (auto& c : calls) {
        emit_li(prog, 5, IMG_BASE + 784 * c.img);
        prog.push_back(enc_cnn(11, 5, c.f3, c.key, c.chal, c.sg));
        prog.push_back(enc_s(2, 6, 11, 0)); prog.push_back(enc_i(0x13, 6, 0, 6, 4));
    }
    prog.push_back(0x00000073);   // ecall

    // ---------- SoC ----------
    sc_time period(1000.0 / mhz, SC_NS);
    sc_clock clk("clk", period);
    sc_signal<bool> auth_req, unlocked, error, cstart, cpriv, cdone, cbusy;
    sc_signal<sc_uint<4>> key, chal, sig, result, fsm, noise; sc_signal<sc_uint<2>> astate; sc_signal<sc_uint<32>> img, pcs;
    RiscvCpu cpu("cpu"); AuthFSM auth("auth"); CnnAccel acc("accel"); LfsrNoise lfsr("lfsr");
    cpu.clk(clk); cpu.auth_req(auth_req); cpu.cnn_start(cstart); cpu.cnn_priv(cpriv); cpu.key(key); cpu.chal(chal); cpu.sig(sig);
    cpu.img_addr(img); cpu.pc_out(pcs); cpu.auth_unlocked(unlocked); cpu.auth_error(error); cpu.cnn_done(cdone); cpu.cnn_result(result);
    auth.clk(clk); auth.req(auth_req); auth.key(key); auth.chal(chal); auth.sig(sig); auth.state(astate); auth.unlocked(unlocked); auth.error(error);
    lfsr.clk(clk); lfsr.noise(noise);
    acc.clk(clk); acc.start(cstart); acc.authorized(unlocked); acc.privacy(cpriv); acc.img_addr(img); acc.noise_in(noise);
    acc.done(cdone); acc.busy(cbusy); acc.result(result); acc.fsm_state(fsm);
    cpu.imem = prog; cpu.dmem = &mem; cpu.bypass = (scen == "bypass"); cpu.period = period;
    auth.secret_key = KEY; auth.hardened = hardened;
    acc.model = &model; acc.mem = &mem; acc.cfg.n_pe = pe; acc.cfg.pipelined = pipe; acc.cfg.shared_mac = !ideal; acc.cfg.clk_mhz = mhz;
    acc.keyed_mask = hardened; acc.klfsr = Lfsr4(KEY);

    sc_trace_file* tf = nullptr;
    if (vcd) {
        tf = sc_create_vcd_trace_file(vcdname.c_str()); tf->set_time_unit(1, SC_NS);
        sc_trace(tf, clk, "clk"); sc_trace(tf, pcs, "cpu_pc"); sc_trace(tf, auth_req, "auth_req"); sc_trace(tf, key, "key"); sc_trace(tf, chal, "challenge");
        sc_trace(tf, sig, "signature"); sc_trace(tf, astate, "auth_state_0IDLE_1VERIFY_2UNLOCK_3ERR"); sc_trace(tf, unlocked, "unlocked"); sc_trace(tf, error, "auth_error");
        sc_trace(tf, cstart, "cnn_start"); sc_trace(tf, cpriv, "privacy_mode"); sc_trace(tf, fsm, "cnn_fsm_state"); sc_trace(tf, cbusy, "cnn_busy");
        sc_trace(tf, cdone, "cnn_done"); sc_trace(tf, noise, "lfsr_noise"); sc_trace(tf, result, "cnn_result");
    }
    sc_start();
    if (tf) sc_close_vcd_trace_file(tf);

    // ---------- report ----------
    double us_per_cycle = 1.0 / mhz;
    printf("=== scenario=%s  PE=%d  %s%s  clk=%.1f MHz%s ===\n", scen.c_str(), pe, pipe ? (ideal ? "pipelined(ideal)" : "pipelined") : "sequential", "", mhz, hardened ? "  [HARDENED]" : "");
    int accepted = 0, correct = 0, invalid = 0, golden_ok = 0, golden_n = 0; long sumcyc = 0;
    Lfsr4 mirror(KEY); int decoded_ok = 0, matchpred = 0; std::set<unsigned> masks;
    size_t ai = 0;
    for (size_t i = 0; i < cpu.traces.size(); i++) {
        auto& t = cpu.traces[i]; const Call& c = calls[i];
        uint32_t r = 0; for (int b = 0; b < 4; b++) r |= (uint32_t)mem[RES_BASE + 4 * i + b] << (8 * b);
        int label = labs[c.img]; std::string how;
        if (t.ok) {
            accepted++; long cyc = acc.cycles_log[ai]; sumcyc += cyc;
            std::array<int32_t, 10> L = acc.logits_log[ai++];
            if (!(custom && c.img == 0)) { golden_n++; if (!memcmp(L.data(), glog + 10 * c.img, 40)) golden_ok++; }
            int pred = argmax10(L.data());
            if (r == (uint32_t)label) correct++; if (r == (uint32_t)pred) matchpred++; if (r >= 10) invalid++;
            masks.insert(r ^ pred);
            if (priv && hardened) { uint32_t d = r ^ mirror.step(); if (d == (uint32_t)pred) decoded_ok++; }
            char b[160]; snprintf(b, 160, "AUTH OK   result=%2u (label %d, clean prediction %d)  accel %ld cycles = %.1f us", r, label, pred, cyc, cyc * us_per_cycle); how = b;
        } else how = r == RES_DENIED ? "AUTH DENIED -> accelerator stayed idle" : r == RES_BYPASS ? "start strobe without auth -> accelerator REFUSED (timeout)" : "error";
        if (!quiet && cpu.traces.size() <= 24) printf("  [%2zu] key=%X chal=%X sig=%X img=%-3d  %s\n", i, c.key, c.chal, c.sg, c.img, how.c_str());
    }
    int tot = (int)cpu.traces.size();
    printf("calls=%d  accepted=%d  denied/blocked=%d  (auth FSM: ok=%ld fail=%ld, accel refused un-authorised starts=%ld)\n", tot, accepted, tot - accepted, auth.n_ok, auth.n_fail, acc.refused);
    if (accepted) {
        double avg = (double)sumcyc / accepted;
        printf("accelerator latency: %.0f cycles = %.4f ms (avg of %d)   CPU instructions retired: %ld\n", avg, avg * us_per_cycle / 1000, accepted, cpu.instret);
        if (golden_n) printf("bit-exact vs Python int8 golden logits: %d/%d identical\n", golden_ok, golden_n);
        if (scen == "normal") printf("accuracy on these images: %d/%d correct\n", correct, accepted);
        if (priv) {
            printf("privacy mode: output == clean prediction in %d/%d cases (chance=1/16), output >= 10 (not even a digit) in %d/%d\n", matchpred, accepted, invalid, accepted);
            printf("distinct XOR masks observed over %d inferences: %zu%s\n", accepted, masks.size(), (masks.size() == 1 && !hardened) ? "  <-- mask is CONSTANT: output is just a fixed relabelling of the class" : "");
            if (hardened) printf("legitimate CPU un-masks keyed LFSR stream: %d/%d predictions recovered\n", decoded_ok, accepted);
        }
    }
    if (scen == "brute") printf("brute force over all 4096 imm12 values: %ld accepted (=%.4f%% ; 1 in %.0f) -- paper claims 2^12 = 4096\n", auth.n_ok, 100.0 * auth.n_ok / 4096, 4096.0 / std::max(1L, auth.n_ok));
    if (scen == "smart") printf("attacker who knows the signature formula but not the key: %ld accepted out of 16 guesses (max 16 tries needed)\n", auth.n_ok);
    if (scen == "replay") printf("replayed captured instruction: %ld/8 accepted\n", auth.n_ok);
    if (layers && !acc.first_sched.empty()) {
        printf("per-layer exposed latency (this configuration):\n"); long prev = 0;
        for (size_t i = 0; i < acc.first_sched.size(); i++) { long e = acc.first_sched[i].end; printf("  %-16s %9ld cycles  %9.4f ms\n", mnist_layers()[i].name, e - prev, (e - prev) / (mhz * 1e3)); prev = e; }
    }
    if (!csv.empty() && accepted) { FILE* f = fopen(csv.c_str(), "a"); TimingCfg c = acc.cfg; fprintf(f, "%d,%s,%ld,%.4f\n", pe, pipe ? (ideal ? "pipelined_ideal" : "pipelined_shared") : "sequential", sumcyc / accepted, sumcyc / accepted / (mhz * 1e3)); fclose(f); }
    return 0;
}
