# Copilot Instructions for Peak Picker Implementation

## Project Context

This project implements a critical component of a 5G NR SSB (Synchronization Signal Block) detection application. Specifically, the peak picker algorithm identifies SSB signals by detecting peaks where the maximum magnitude squared of the PSS (Primary Synchronization Signal) correlation (`xcorr`) within a windowed sequence exceeds a predefined `threshold`. The reference implementation is provided in MATLAB (`peakPicker.m`) along with a testbench (`peakPicker_tb.m`).

## Task Description

Your task is to translate the MATLAB peak picker algorithm into efficient HLS C++ code while preserving exact functionality. The implementation should be optimized for FPGA deployment using Xilinx HLS directives.

**Required Files:**

- `peakPicker.hpp`: Header file with type definitions, function declarations, and parameters
- `peakPicker.cpp`: Implementation file with the core algorithm
- `peakPicker_tb.cpp`: C++ testbench that validates the implementation against reference data

## Input/Output Specifications

- **Inputs:**
  - PSS correlation magnitude squared values (fixed-point stream)
  - Threshold value (fixed-point)
- **Outputs:**
  - Peak locations (integer indices)

## Implementation Requirements

### Functional Requirements

- Implement the `peakPicker()` function in HLS C++ with exactly the same behavior as the MATLAB reference
- Follow bit-accurate implementation of the MATLAB algorithm (results must match reference within specified error margins)
- Document code thoroughly with comments explaining the algorithm and optimization decisions

### Interface and Data Type Requirements

- Use `hls::stream` interfaces with appropriate buffer depths for streaming data
- Implement fixed-point arithmetic with `ap_fixed<W,I>` (specify exact bit widths based on precision requirements)
- Use `hls::complex<ap_fixed<W,I>>` for any complex number operations
- Define all constant parameters in `peakPicker.hpp` using `#define` or `constexpr`
- Create descriptive type aliases with `typedef` or `using` statements

### File I/O and Validation only in testbench file `peakPicker_tb.cpp`

- Read input PSS correlation data from `pssCorrMagSq_#_in.txt` (one float value per line)
- Read threshold values from `threshold_in.txt` (one float value per line)
- Read reference peak locations from `locations_#_ref.txt` (one integer value per line)
- Implement robust error checking for file operations with clear error messages
- Calculate and report both absolute and relative errors between your implementation and reference values

### Performance Optimization

- Apply `#pragma HLS PIPELINE II=1` to performance-critical loops
- Use `#pragma HLS DATAFLOW` for task-level pipelining
- Implement arrays exceeding 64 elements using dual-port block RAM
- Apply memory partitioning to arrays requiring parallel access
- Configure optimization directives based on throughput requirements
- Balance resource usage and performance based on target FPGA constraints

### Coding Style

- Define all constants, types, and function declarations in `peakPicker.hpp`
- Implement core algorithm in `peakPicker.cpp` with consistent style
- Follow naming convention: camelCase for variables, PascalCase for types
- Use self-documenting identifier names that clearly reflect their purpose

## Deliverables

- Fully commented HLS C++ implementation files
- Comprehensive testbench demonstrating functional correctness
- Description of optimization approaches and their impact on performance
