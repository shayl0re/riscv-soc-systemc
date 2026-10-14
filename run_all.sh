#!/bin/bash
# Reproduce everything (WSL2 Ubuntu / Linux). Run from the project folder:  bash run_all.sh
set -e
export SYSTEMC_HOME=${SYSTEMC_HOME:-$HOME/systemc_install}
cd systemc && make SYSTEMC_HOME=$SYSTEMC_HOME && cd ..
echo "== 1. PE unit test ==";           (cd systemc && ./tb_pe | grep -E "PASS|FAIL")
echo "== 2. normal inference ==";       (cd systemc && ./soc_sim --scenario normal --n 5 --pe 128 --layers | sed -n '/===/,$p')
for s in wrongkey wrongsig tamper bypass smart; do echo "== attack: $s =="; (cd systemc && ./soc_sim --scenario $s --n 4 | sed -n '/===/,$p'); done
echo "== replay (paper) ==";            (cd systemc && ./soc_sim --scenario replay --quiet | tail -2)
echo "== replay (hardened) ==";         (cd systemc && ./soc_sim --scenario replay --hardened --quiet | tail -2)
echo "== brute force ==";               (cd systemc && ./soc_sim --scenario brute --quiet | tail -2)
echo "== privacy (paper) ==";           (cd systemc && ./soc_sim --scenario privacy --n 20 --quiet | tail -3)
echo "== privacy (hardened) ==";        (cd systemc && ./soc_sim --scenario privacy --n 20 --hardened --quiet | tail -3)
echo "== latency table ==";             (cd systemc && ./soc_sim --table | grep TOTAL)
python3 python/analysis.py > /dev/null && echo "results/ updated (tables + figures)"
