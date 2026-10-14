"""Runs the SystemC simulator, compares with the paper, produces tables + figures in ../results"""
import subprocess, re, json, os, csv, io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "..", "systemc"); OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
def run(*a):
    return subprocess.run(["./soc_sim", *a], cwd=SC, capture_output=True, text=True).stdout

# ---------------- latency table from the simulator ----------------
rows = list(csv.DictReader(io.StringIO(run("--table"))))
def tot(pe, mode): return float([r for r in rows if r["pe"] == str(pe) and r["mode"] == mode and r["layer"] == "TOTAL"][0]["ms"])
def layer(pe, mode, name): return float([r for r in rows if r["pe"] == str(pe) and r["mode"] == mode and r["layer"].startswith(name)][0]["ms"])
def grouped(pe, mode):  # same 5 groups as paper Table I (Dense = Dense1+Dense2+Argmax)
    return [layer(pe, mode, "Conv 3x3x32"), layer(pe, mode, "MaxPool 2x2x32"), layer(pe, mode, "Conv 3x3x64"), layer(pe, mode, "MaxPool 2x2x64"),
            layer(pe, mode, "Dense 1600") + layer(pe, mode, "Dense 128") + layer(pe, mode, "Argmax")]
# paper Table I (ms): [conv1, pool1, conv2, pool2, dense, total]
PAPER = {(9, "seq"): [1.45, .41, 15.72, .11, 1.31, 19.01], (9, "pipe"): [.32, .09, 3.45, .03, .30, 4.18],
         (128, "seq"): [.12, .56, .81, .08, 1.14, 2.71], (128, "pipe"): [.02, .06, .08, .01, .08, .25]}
MODE = {"seq": "sequential", "pipe": "pipelined_ideal"}
names = ["Conv1", "Pool1", "Conv2", "Pool2", "Dense"]

md = ["# Validation: SystemC simulation vs paper Table I (MNIST, ms)\n", "Clock assumed 15 MHz (back-computed from paper's 9-PE sequential total; paper does not state it).\n",
      "| PEs | mode | layer | paper | simulated | error % |", "|---|---|---|---|---|---|"]
for (pe, m), pv in PAPER.items():
    sv = grouped(pe, MODE[m]) + [tot(pe, MODE[m])]
    for nme, p, s in zip(names + ["TOTAL"], pv, sv):
        md.append(f"| {pe} | {m} | {nme} | {p:.2f} | {s:.3f} | {100 * (s - p) / p:+.0f} |")
md.append("")
sp = {k: (tot(k[0], "sequential"), tot(k[0], MODE[k[1]])) for k in [(9, "pipe"), (128, "pipe")]}
md += ["## Speed-up of pipelining (same PE count)\n", "| PEs | paper | simulated (ideal, dedicated MAC array per layer) | simulated (shared MAC array, paper Fig.1) |", "|---|---|---|---|"]
for pe, pp in [(9, 19.01 / 4.18), (128, 2.71 / 0.25)]:
    md.append(f"| {pe} | {pp:.2f}x | {tot(pe, 'sequential') / tot(pe, 'pipelined_ideal'):.2f}x | {tot(pe, 'sequential') / tot(pe, 'pipelined_shared'):.2f}x |")
md.append(f"\nPE scaling, sequential: 1->9 PE = {tot(1,'sequential')/tot(9,'sequential'):.1f}x, 9->128 PE = {tot(9,'sequential')/tot(128,'sequential'):.1f}x (paper 9->128: {19.01/2.71:.1f}x). 1 PE = {tot(1,'sequential'):.1f} ms.\n")

# Amdahl-style bound: single-image latency can never beat the slowest layer
macs = [194688, 2230272, 204800, 1280]; share = macs[1] / sum(macs)
md += ["## Consistency checks on the paper's own numbers\n",
       f"* Conv2 holds {100*share:.0f}% of all MACs ({sum(macs):,} MACs/inference; paper's '2.64M ops' counts one MAC as one op). Layer pipelining cannot make single-image latency smaller than the conv2 time, so it cannot give 10x unless conv2 itself gets more hardware.",
       f"* Paper's MNIST speedup from its own Table I is 2.71/0.25 = {2.71/0.25:.2f}x, but the text states 10.68x (CIFAR-10: 162.77/15.93 = {162.77/15.93:.2f}x, matches the text).",
       "* 128 PEs peak = 128 MAC/cycle. Paper's pipelined conv2 (0.08 ms) needs " + f"{2230272/128/0.08e-3/1e6:.0f} MHz even at 100% utilisation; whole network (0.25 ms) needs >= {sum(macs)/128/0.25e-3/1e6:.0f} MHz.",
       f"* Paper reports 29 DSP48E1 for the 128-PE MNIST design (Table III). 29 DSP x 100 MHz = 2.9 GMAC/s, below the claimed 10.56 GOPS (needs LUT multipliers or >= {10.56e9/29/1e6:.0f} MHz).",
       f"* Unit slip in Table V: 10.56 GOPS / 0.175 W = {10.56/0.175:.1f} GOPS/W (= {10.56/0.175/1000:.3f} TOPS/W), yet the paper reports 60.3 TOPS/W. Same for CIFAR-10 (2.48 GOPS / 0.241 W = {2.48/0.241:.1f} GOPS/W vs 10.3 'TOPS/W') and for area efficiency (10.56 GOPS / 0.088 mm2 = {10.56/0.088:.0f} GOPS/mm2 vs '120 TOPS/mm2'). Energy 0.0438 mJ = 175 mW x 0.25 ms is consistent.", ""]

# ---------------- security numbers ----------------
b = run("--scenario", "brute", "--quiet", "--pe", "128"); smart = run("--scenario", "smart", "--quiet")
rp = run("--scenario", "replay", "--quiet"); rph = run("--scenario", "replay", "--quiet", "--hardened")
pv = run("--scenario", "privacy", "--n", "50", "--quiet"); pvh = run("--scenario", "privacy", "--n", "50", "--quiet", "--hardened")
nrm = run("--scenario", "normal", "--n", "100", "--quiet")
g = lambda pat, s: re.search(pat, s).groups()
acc_ok = int(g(r"accepted=(\d+)", b)[0]); smart_ok = int(g(r"accepted=(\d+)", smart)[0])
replay_ok = int(g(r"accepted=(\d+)", rp)[0]); replay_h = int(g(r"accepted=(\d+)", rph)[0])
priv_match = g(r"clean prediction in (\d+)/(\d+)", pv); priv_inv = g(r"digit\) in (\d+)/", pv); masks = int(g(r"masks observed over \d+ inferences: (\d+)", pv)[0])
dec = g(r"recovered|(\d+)/(\d+) predictions", pvh) if False else re.search(r"(\d+)/(\d+) predictions recovered", pvh).groups()
gold = g(r"identical|(\d+)/(\d+) identical", nrm) if False else re.search(r"(\d+)/(\d+) identical", nrm).groups()
acc = re.search(r"accuracy on these images: (\d+)/(\d+)", nrm).groups()
md += ["# Security experiments (all run through the SystemC SoC)\n",
       f"* Bit-exact check, 100 images: {gold[0]}/{gold[1]} identical logits vs Python int8 reference; accuracy {acc[0]}/{acc[1]}.",
       f"* Brute force over all 4096 imm12 values: {acc_ok} accepted -> 1 in {4096//acc_ok} (paper claims 2^12 = 4096 attempts). Cause: attacker may pick ANY challenge; only key and signature must match (8 effective bits).",
       f"* Attacker knowing the formula but not the key: {smart_ok} accepted in 16 tries (worst case 16 tries).",
       f"* Replay of one captured valid instruction (8 times): paper design accepts {replay_ok}/8; hardened (freshness check) accepts {replay_h}/8.",
       f"* Privacy mode (paper): output equals the clean prediction in {priv_match[0]}/{priv_match[1]} runs (the LFSR is never 0, so the true class is never output); {priv_inv[0]}/{priv_match[1]} outputs are not even valid digits (>=10); {masks} distinct masks seen. Legitimate user cannot decode it.",
       f"* Privacy mode (hardened, keyed LFSR seed shared with CPU): legitimate CPU recovers {dec[0]}/{dec[1]} predictions.", ""]
open(os.path.join(OUT, "results.md"), "w").write("\n".join(md)); print("\n".join(md))
json.dump({"brute_accepted": acc_ok, "smart": smart_ok, "replay": replay_ok, "replay_hard": replay_h}, open(os.path.join(OUT, "security.json"), "w"))

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
# 1 latency scaling
fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(3); w = 0.26
for i, (m, lab, c) in enumerate([("sequential", "sequential", "#4C72B0"), ("pipelined_shared", "pipelined (shared MAC array)", "#55A868"), ("pipelined_ideal", "pipelined (ideal bound)", "#C44E52")]):
    v = [tot(p, m) for p in (1, 9, 128)]; ax.bar(x + (i - 1) * w, v, w, label=lab, color=c)
ax.scatter([1 - w, 2 - w, 1 + w, 2 + w], [19.01, 2.71, 4.18, 0.25], marker="D", color="k", zorder=5, label="paper (Table I)")
ax.set_yscale("log"); ax.set_xticks(x, ["1 PE", "9 PEs", "128 PEs"]); ax.set_ylabel("MNIST latency (ms, log)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Latency vs PE count (SystemC model, 15 MHz)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_latency_scaling.png"), dpi=200); plt.close()
# 2 validation per layer
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, pe in zip(axs, (9, 128)):
    x = np.arange(5); ax.bar(x - .2, PAPER[(pe, "seq")][:5], .4, label="paper", color="#999"); ax.bar(x + .2, grouped(pe, "sequential"), .4, label="simulated", color="#4C72B0")
    ax.set_xticks(x, names); ax.set_ylabel("ms"); ax.set_title(f"{pe} PEs, sequential"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_validation.png"), dpi=200); plt.close()
# 3 stacked layer breakdown
fig, ax = plt.subplots(figsize=(8, 3.6)); labs = ["Seq 128PE", "Pipelined-ideal 128PE", "Paper pipelined 128PE (claim)"]
cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]; data = [grouped(128, "sequential"), grouped(128, "pipelined_ideal"), PAPER[(128, "pipe")][:5]]
for i, d in enumerate(data):
    left = 0
    for j, v in enumerate(d): ax.barh(i, v, left=left, color=cols[j], label=names[j] if i == 0 else None); left += v
ax.set_yticks(range(3), labs); ax.set_xlabel("ms"); ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.18)); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pipeline_bound.png"), dpi=200); plt.close()
# 4 security
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].bar(["paper claim\n(2^12)", "random guess\n(simulated)", "knows formula\n(simulated)"], [4096, 4096 // max(acc_ok, 1), 16], color=["#999", "#C44E52", "#C44E52"]); axs[0].set_yscale("log"); axs[0].set_ylabel("attempts needed"); axs[0].set_title("Brute-force effort")
axs[1].bar(["paper design", "hardened"], [replay_ok, replay_h], color=["#C44E52", "#55A868"]); axs[1].set_ylabel("replayed instrs accepted (of 8)"); axs[1].set_title("Replay attack")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_security.png"), dpi=200); plt.close()
print("figures written")
