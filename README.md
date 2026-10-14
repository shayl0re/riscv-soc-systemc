# Secure Edge-AI RISC-V SoC: SystemC Simulation

A complete software simulation of a secure, privacy-aware System-on-Chip for edge AI.

**Core features:**
- ✅ RV32IM CPU with custom CNN instruction (opcode 0x7B)
- ✅ 8-layer int8 CNN accelerator (1, 9, or 128 PE configs)
- ✅ Challenge-response authentication FSM
- ✅ LFSR output privacy mode
- ✅ Full test suite: validation, security attacks
- ✅ Live demo GUI (draw digit → classify)
- ✅ Verilog RTL for synthesis

## Quick Start

```bash
# From systemc/:
./soc_sim --scenario normal --n 5 --pe 128
./soc_sim --pe 1 / 9 / 128
./soc_sim --scenario privacy --n 10
./soc_sim --scenario wrongkey --n 3
```

```bash
# From python/:
python3 demo_app.py   # Live GUI
python3 demo_app.py --cli 5  # CLI
```

## Results

| PE count | Sequential | Pipelined (ideal) |
|----------|------------|------------------|
| 1 PE | 175.89 ms | 163.55 ms |
| 9 PEs | 19.98 ms | 18.28 ms |
| 128 PEs | 2.84 ms | 2.62 ms |

**Accuracy:** 99% on 200 test images  
**Security:** 16/16 attacks blocked

## GitHub

Code: https://github.com/shayl0re/riscv-soc-systemc

