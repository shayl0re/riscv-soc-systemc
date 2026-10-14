# Validation: SystemC simulation vs paper Table I (MNIST, ms)

Clock assumed 15 MHz (back-computed from paper's 9-PE sequential total; paper does not state it).

| PEs | mode | layer | paper | simulated | error % |
|---|---|---|---|---|---|
| 9 | seq | Conv1 | 1.45 | 1.443 | -0 |
| 9 | seq | Pool1 | 0.41 | 0.362 | -12 |
| 9 | seq | Conv2 | 15.72 | 16.522 | +5 |
| 9 | seq | Pool2 | 0.11 | 0.108 | -2 |
| 9 | seq | Dense | 1.31 | 1.531 | +17 |
| 9 | seq | TOTAL | 19.01 | 19.979 | +5 |
| 9 | pipe | Conv1 | 0.32 | 1.443 | +351 |
| 9 | pipe | Pool1 | 0.09 | 0.028 | -69 |
| 9 | pipe | Conv2 | 3.45 | 15.245 | +342 |
| 9 | pipe | Pool2 | 0.03 | 0.022 | -28 |
| 9 | pipe | Dense | 0.30 | 1.531 | +410 |
| 9 | pipe | TOTAL | 4.18 | 18.282 | +337 |
| 128 | seq | Conv1 | 0.12 | 0.207 | +73 |
| 128 | seq | Pool1 | 0.56 | 0.027 | -95 |
| 128 | seq | Conv2 | 0.81 | 2.361 | +192 |
| 128 | seq | Pool2 | 0.08 | 0.009 | -89 |
| 128 | seq | Dense | 1.14 | 0.222 | -80 |
| 128 | seq | TOTAL | 2.71 | 2.840 | +5 |
| 128 | pipe | Conv1 | 0.02 | 0.207 | +937 |
| 128 | pipe | Pool1 | 0.06 | 0.002 | -96 |
| 128 | pipe | Conv2 | 0.08 | 2.174 | +2618 |
| 128 | pipe | Pool2 | 0.01 | 0.002 | -82 |
| 128 | pipe | Dense | 0.08 | 0.222 | +178 |
| 128 | pipe | TOTAL | 0.25 | 2.621 | +948 |

## Speed-up of pipelining (same PE count)

| PEs | paper | simulated (ideal, dedicated MAC array per layer) | simulated (shared MAC array, paper Fig.1) |
|---|---|---|---|
| 9 | 4.55x | 1.09x | 1.02x |
| 128 | 10.84x | 1.08x | 1.01x |

PE scaling, sequential: 1->9 PE = 8.8x, 9->128 PE = 7.0x (paper 9->128: 7.0x). 1 PE = 175.9 ms.

## Consistency checks on the paper's own numbers

* Conv2 holds 85% of all MACs (2,631,040 MACs/inference; paper's '2.64M ops' counts one MAC as one op). Layer pipelining cannot make single-image latency smaller than the conv2 time, so it cannot give 10x unless conv2 itself gets more hardware.
* Paper's MNIST speedup from its own Table I is 2.71/0.25 = 10.84x, but the text states 10.68x (CIFAR-10: 162.77/15.93 = 10.22x, matches the text).
* 128 PEs peak = 128 MAC/cycle. Paper's pipelined conv2 (0.08 ms) needs 218 MHz even at 100% utilisation; whole network (0.25 ms) needs >= 82 MHz.
* Paper reports 29 DSP48E1 for the 128-PE MNIST design (Table III). 29 DSP x 100 MHz = 2.9 GMAC/s, below the claimed 10.56 GOPS (needs LUT multipliers or >= 364 MHz).
* Unit slip in Table V: 10.56 GOPS / 0.175 W = 60.3 GOPS/W (= 0.060 TOPS/W), yet the paper reports 60.3 TOPS/W. Same for CIFAR-10 (2.48 GOPS / 0.241 W = 10.3 GOPS/W vs 10.3 'TOPS/W') and for area efficiency (10.56 GOPS / 0.088 mm2 = 120 GOPS/mm2 vs '120 TOPS/mm2'). Energy 0.0438 mJ = 175 mW x 0.25 ms is consistent.

# Security experiments (all run through the SystemC SoC)

* Bit-exact check, 100 images: 100/100 identical logits vs Python int8 reference; accuracy 98/100.
* Brute force over all 4096 imm12 values: 16 accepted -> 1 in 256 (paper claims 2^12 = 4096 attempts). Cause: attacker may pick ANY challenge; only key and signature must match (8 effective bits).
* Attacker knowing the formula but not the key: 1 accepted in 16 tries (worst case 16 tries).
* Replay of one captured valid instruction (8 times): paper design accepts 8/8; hardened (freshness check) accepts 1/8.
* Privacy mode (paper): output equals the clean prediction in 0/50 runs (the LFSR is never 0, so the true class is never output); 20/50 outputs are not even valid digits (>=10); 15 distinct masks seen. Legitimate user cannot decode it.
* Privacy mode (hardened, keyed LFSR seed shared with CPU): legitimate CPU recovers 50/50 predictions.
