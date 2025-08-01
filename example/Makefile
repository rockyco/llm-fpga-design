# Makefile for HLS Project

# Set the design name
DESIGN_NAME = peakPicker

# Configuration variables
CSIM = 1
CSYNTH = 1
COSIM = 1
EXPORT_IP = 1
VIVADO_IMPL = 1

# Hardware configuration
CLOCK_FREQ = 256
FPGA_PART = xc7k410t-ffg900-2
CLOCK_UNCERTAINTY = 12.5

# Vitis HLS installation path - modify this to match your installation
VITIS_HLS_PATH ?= /opt/Xilinx/Vitis_HLS/2024.2
# HLS compiler and flags - use full path to vitis_hls executable
HLS = $(VITIS_HLS_PATH)/bin/vitis_hls

# You can also set VITIS_HLS_PATH via environment variable before running make:
# export VITIS_HLS_PATH=/path/to/your/Vitis_HLS/installation

# Optional: Uncomment to source Vitis HLS settings before each HLS command
# HLS = source $(VITIS_HLS_PATH)/settings64.sh && $(VITIS_HLS_PATH)/bin/vitis_hls

HLS_PROJECT = proj_$(DESIGN_NAME)
HLS_SOLUTION = solution1

# C++ compiler and flags for testbench
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17
INCLUDES = -I$(XILINX_HLS)/include

# Source files
SRC_FILES = $(DESIGN_NAME).cpp
TB_FILES = $(DESIGN_NAME)_tb.cpp
TEST_DATA_DIR = .
# Get a list of all text files in the data directory
TEST_DATA_FILES := $(wildcard $(TEST_DATA_DIR)/*.txt)

# Target names
TB_EXE = $(DESIGN_NAME)_test
CSIM_TCL = csim.tcl
CSYNTH_TCL = csynth.tcl
COSIM_TCL = cosim.tcl
EXPORT_TCL = export.tcl
IMPL_TCL = impl.tcl

# Calculate clock period in ns from MHz
CLOCK_PERIOD := $(shell echo "scale=2; 1000 / $(CLOCK_FREQ)" | bc)

.PHONY: all clean tb csim csynth cosim export_ip impl help

all: clean tb csim csynth

# Standalone testbench using GCC
tb:
	@echo "Building standalone testbench..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC_FILES) $(TB_FILES) -o $(TB_EXE)
	@echo "Running standalone testbench..."
	./$(TB_EXE)

# HLS C Simulation
csim:
ifeq ($(CSIM), 1)
	@echo "Running HLS C Simulation..."
	@echo "open_project $(HLS_PROJECT)" > $(CSIM_TCL)
	@echo "set_top peakPicker" >> $(CSIM_TCL)
	@echo "add_files $(SRC_FILES)" >> $(CSIM_TCL)
	@echo "add_files -tb $(TB_FILES)" >> $(CSIM_TCL)
	@for file in $(TEST_DATA_FILES); do \
		echo "add_files -tb $$file" >> $(CSIM_TCL); \
	done
	@echo "open_solution $(HLS_SOLUTION)" >> $(CSIM_TCL)
	@echo "set_part {$(FPGA_PART)}" >> $(CSIM_TCL)
	@echo "csim_design" >> $(CSIM_TCL)
	@echo "exit" >> $(CSIM_TCL)
	$(HLS) -f $(CSIM_TCL)
endif

# HLS C Synthesis
csynth:
ifeq ($(CSYNTH), 1)
	@echo "Running HLS C Synthesis..."
	@echo "open_project $(HLS_PROJECT)" > $(CSYNTH_TCL)
	@echo "set_top peakPicker" >> $(CSYNTH_TCL)
	@echo "add_files $(SRC_FILES)" >> $(CSYNTH_TCL)
	@echo "add_files -tb $(TB_FILES)" >> $(CSYNTH_TCL)
	@for file in $(TEST_DATA_FILES); do \
		echo "add_files -tb $$file" >> $(CSYNTH_TCL); \
	done
	@echo "open_solution $(HLS_SOLUTION)" >> $(CSYNTH_TCL)
	@echo "set_part {$(FPGA_PART)}" >> $(CSYNTH_TCL)
	@echo "create_clock -period $(CLOCK_PERIOD) -name default" >> $(CSYNTH_TCL)
	@echo "set_clock_uncertainty $(CLOCK_UNCERTAINTY)%" >> $(CSYNTH_TCL)
	@echo "csynth_design" >> $(CSYNTH_TCL)
	@echo "exit" >> $(CSYNTH_TCL)
	$(HLS) -f $(CSYNTH_TCL)
endif

# HLS C/RTL Co-simulation
cosim:
ifeq ($(COSIM), 1)
	@echo "Running HLS C/RTL Co-simulation..."
	@echo "open_project $(HLS_PROJECT)" > $(COSIM_TCL)
	@echo "set_top peakPicker" >> $(COSIM_TCL)
	@echo "add_files $(SRC_FILES)" >> $(COSIM_TCL)
	@echo "add_files -tb $(TB_FILES)" >> $(COSIM_TCL)
	@for file in $(TEST_DATA_FILES); do \
		echo "add_files -tb $$file" >> $(COSIM_TCL); \
	done
	@echo "open_solution $(HLS_SOLUTION)" >> $(COSIM_TCL)
	@echo "set_part {$(FPGA_PART)}" >> $(COSIM_TCL)
	@echo "create_clock -period $(CLOCK_PERIOD) -name default" >> $(COSIM_TCL)
	@echo "set_clock_uncertainty $(CLOCK_UNCERTAINTY)%" >> $(COSIM_TCL)
	@echo "cosim_design" >> $(COSIM_TCL)
	@echo "exit" >> $(COSIM_TCL)
	$(HLS) -f $(COSIM_TCL)
endif

# Export RTL as IP
export_ip:
ifeq ($(EXPORT_IP), 1)
	@echo "Exporting IP..."
	@echo "open_project $(HLS_PROJECT)" > $(EXPORT_TCL)
	@echo "set_top peakPicker" >> $(EXPORT_TCL)
	@echo "add_files $(SRC_FILES)" >> $(EXPORT_TCL)
	@echo "add_files -tb $(TB_FILES)" >> $(EXPORT_TCL)
	@for file in $(TEST_DATA_FILES); do \
		echo "add_files -tb $$file" >> $(EXPORT_TCL); \
	done
	@echo "open_solution $(HLS_SOLUTION)" >> $(EXPORT_TCL)
	@echo "set_part {$(FPGA_PART)}" >> $(EXPORT_TCL)
	@echo "create_clock -period $(CLOCK_PERIOD) -name default" >> $(EXPORT_TCL)
	@echo "set_clock_uncertainty $(CLOCK_UNCERTAINTY)%" >> $(EXPORT_TCL)
	@echo "export_design -format ip_catalog" >> $(EXPORT_TCL)
	@echo "exit" >> $(EXPORT_TCL)
	$(HLS) -f $(EXPORT_TCL)
endif

# Run Implementation in Vivado
impl:
ifeq ($(VIVADO_IMPL), 1)
	@echo "Running Vivado Implementation..."
	@echo "open_project $(HLS_PROJECT)" > $(IMPL_TCL)
	@echo "set_top peakPicker" >> $(IMPL_TCL)
	@echo "add_files $(SRC_FILES)" >> $(IMPL_TCL)
	@echo "add_files -tb $(TB_FILES)" >> $(IMPL_TCL)
	@for file in $(TEST_DATA_FILES); do \
		echo "add_files -tb $$file" >> $(IMPL_TCL); \
	done
	@echo "open_solution $(HLS_SOLUTION)" >> $(IMPL_TCL)
	@echo "set_part {$(FPGA_PART)}" >> $(IMPL_TCL)
	@echo "create_clock -period $(CLOCK_PERIOD) -name default" >> $(IMPL_TCL)
	@echo "set_clock_uncertainty $(CLOCK_UNCERTAINTY)%" >> $(IMPL_TCL)
	@echo "export_design -flow impl" >> $(IMPL_TCL)
	@echo "exit" >> $(IMPL_TCL)
	$(HLS) -f $(IMPL_TCL)
endif

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(HLS_PROJECT) *.dat *.log *.tcl $(TB_EXE) *.o *.out

# Help information
help:
	@echo "Makefile for HLS Peak Picker Project"
	@echo ""
	@echo "Targets:"
	@echo "  all        : Run clean, testbench, C simulation, and C synthesis"
	@echo "  tb         : Build and run standalone testbench"
	@echo "  csim       : Run HLS C simulation"
	@echo "  csynth     : Run HLS C synthesis"
	@echo "  cosim      : Run HLS C/RTL co-simulation"
	@echo "  export_ip  : Export RTL as IP catalog"
	@echo "  impl       : Run implementation in Vivado"
	@echo "  clean      : Clean up generated files"
	@echo "  help       : Display this help information"
	@echo ""
	@echo "Configuration (set to 1 to enable, 0 to disable):"
	@echo "  CSIM       = $(CSIM)  (C simulation)"
	@echo "  CSYNTH     = $(CSYNTH)  (C synthesis)"
	@echo "  COSIM      = $(COSIM)  (C/RTL co-simulation)"
	@echo "  EXPORT_IP  = $(EXPORT_IP)  (Export IP)"
	@echo "  VIVADO_IMPL= $(VIVADO_IMPL)  (Vivado implementation)"
	@echo ""
	@echo "Hardware Configuration:"
	@echo "  CLOCK_FREQ = $(CLOCK_FREQ)MHz (Clock frequency)"
	@echo "  CLOCK_PERIOD = $(CLOCK_PERIOD)ns (Clock period)"
	@echo "  FPGA_PART  = $(FPGA_PART)  (FPGA part)"
	@echo "  CLOCK_UNCERTAINTY = $(CLOCK_UNCERTAINTY)%  (Clock uncertainty)"
