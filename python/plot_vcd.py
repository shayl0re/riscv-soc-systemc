"""Render key signals of the SystemC VCD files as PNG (use GTKWave for the interactive view)."""
import os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse(path):
    ids, vals, t = {}, {}, 0
    for line in open(path):
        line = line.strip()
        m = re.match(r"\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)", line)
        if m: ids[m[2]] = (m[3], int(m[1])); vals[m[3]] = [(0, 0)]; continue
        if line.startswith("#"): t = int(line[1:]); continue
        if line.startswith("b"):
            b, i = line[1:].split(); 
            if i in ids: vals[ids[i][0]].append((t, int(b.replace("x", "0").replace("z", "0"), 2)))
        elif line and line[0] in "01xz" and line[1:] in ids: vals[ids[line[1:]][0]].append((t, 1 if line[0] == "1" else 0))
    return vals

def draw(vals, sigs, t0, t1, title, out, unit="ns", scale=1.0):
    fig, axs = plt.subplots(len(sigs), 1, figsize=(10, 1.0 * len(sigs) + 0.6), sharex=True)
    for ax, (name, label, hexa) in zip(axs, sigs):
        s = [(t, v) for t, v in vals[name]]; xs, ys = [], []
        for i, (t, v) in enumerate(s):
            xs.append(t); ys.append(v)
            if i + 1 < len(s): xs.append(s[i + 1][0]); ys.append(v)
        xs.append(t1); ys.append(ys[-1])
        ax.step([x * scale for x in xs], ys, where="post", color="#1f4e79", lw=1.4)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8); ax.set_yticks([])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if hexa:
            last = None
            for t, v in s:
                if t0 <= t <= t1 and v != last: ax.text(t * scale, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .35, f"{v:X}", fontsize=8, color="#c00000"); last = v
    axs[0].set_title(title, fontsize=10); axs[-1].set_xlabel(f"time ({unit})"); axs[-1].set_xlim(t0 * scale, t1 * scale)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close()

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "waveforms"); R = os.path.join(W, "..")
sig_auth = [("auth_req", "auth_req", False), ("key", "key", True), ("challenge", "challenge", True), ("signature", "signature", True),
            ("auth_state_0IDLE_1VERIFY_2UNLOCK_3ERR", "auth FSM state", True), ("unlocked", "unlocked", False), ("auth_error", "auth_error", False),
            ("cnn_start", "cnn_start", False), ("cnn_busy", "cnn_busy", False), ("cnn_fsm_state", "CNN FSM state", True)]
v = parse(os.path.join(W, "soc_wave.vcd")); draw(v, sig_auth, 0, 1500, "Authorised call: AUTH FSM IDLE -> VERIFY -> UNLOCKED, then CNN starts", os.path.join(R, "wave_auth_ok.png"))
v = parse(os.path.join(W, "wave_denied.vcd")); draw(v, sig_auth, 0, 1500, "Wrong key: AUTH FSM IDLE -> VERIFY -> ERROR, CNN never starts", os.path.join(R, "wave_auth_denied.png"))
v = parse(os.path.join(W, "soc_wave.vcd"))
draw(v, [("cnn_fsm_state", "CNN FSM state\n(2 Conv1 3 Pool1 4 Conv2\n5 Pool2 6 Dense1 7 Dense2\n8 Argmax 9 Done)", True), ("cnn_busy", "cnn_busy", False), ("cnn_done", "cnn_done", False), ("cnn_result", "result", True)],
     0, 2_900_000, "One full MNIST inference, 128 PEs, 15 MHz (accelerator FSM)", os.path.join(R, "wave_inference.png"), unit="ms", scale=1e-6)
v = parse(os.path.join(W, "pe_wave.vcd"))
draw(v, [("clk", "clk", False), ("a", "a", True), ("b", "b", True), ("valid", "valid", False), ("clear", "clear", False), ("stage1_product", "stage-1 product", True), ("stage2_acc", "stage-2 acc", True)],
     0, 260, "MAC PE: one multiply-accumulate per clock, 2-cycle pipeline fill", os.path.join(R, "wave_mac_pe.png"))
print("ok")
