# Usage: make SYSTEMC_HOME=/path/to/systemc/install     (default below)
SYSTEMC_HOME ?= /home/claude/sc_install
CXX ?= g++
CXXFLAGS = -std=c++17 -O2 -Wno-deprecated-declarations -I$(SYSTEMC_HOME)/include -Isrc
LDFLAGS = -L$(SYSTEMC_HOME)/lib -L$(SYSTEMC_HOME)/lib64 -lsystemc -lpthread -Wl,-rpath,$(SYSTEMC_HOME)/lib:$(SYSTEMC_HOME)/lib64
all: soc_sim tb_pe
soc_sim: src/soc_main.cpp src/*.h
	$(CXX) $(CXXFLAGS) src/soc_main.cpp -o $@ $(LDFLAGS)
tb_pe: src/tb_pe.cpp src/pe.h
	$(CXX) $(CXXFLAGS) src/tb_pe.cpp -o $@ $(LDFLAGS)
clean:
	rm -f soc_sim tb_pe *.vcd
