#!/bin/bash
# Synthesis of the SystemC-equivalent RTL blocks with Yosys (Xilinx 7-series mapping). Not Vivado.
cd "$(dirname "$0")"; mkdir -p syn
run() { name=$1; top=$2; files=$3; opts=$4; extra=$5
  yosys -p "read_verilog $files; $extra hierarchy -top $top; synth_xilinx -family xc7 $opts -top $top; tee -o syn/$name.stat stat" > syn/$name.log 2>&1; }
run auth_fsm   auth_fsm   auth_fsm.v
run lfsr_noise lfsr_noise lfsr_noise.v
run secure_gate secure_gate "auth_fsm.v lfsr_noise.v secure_gate.v"
run maxpool_pe maxpool_pe maxpool_pe.v
run mac_pe     mac_pe     mac_pe.v
run mac_pe_nodsp mac_pe   mac_pe.v "-nodsp"
for n in 9 128; do
  run mac_array_$n mac_array "mac_pe.v mac_array.v" "" "chparam -set N $n mac_array;"
  run mac_array_${n}_nodsp mac_array "mac_pe.v mac_array.v" "-nodsp" "chparam -set N $n mac_array;"
done
