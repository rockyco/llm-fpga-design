# Copilot Instructions for {component} Implementation

## Project Context

This project implements a component for an FPGA-based signal processing application. The {component} algorithm needs to be translated from a reference implementation to efficient HLS C++ code for FPGA deployment.

## Task Description

Your task is to translate the reference {component} algorithm into efficient HLS C++ code while preserving exact functionality. The implementation should be optimized for FPGA deployment using Xilinx HLS directives.

**Required Files:**

- `{component}.hpp`: Header file with type definitions, function declarations, and parameters
- `{component}.cpp`: Implementation file with the core algorithm
- `{component}_tb.cpp`: C++ testbench that validates the implementation against reference data

## Input/Output Specifications

- **Inputs:**
  - [To be specified based on {component} requirements]
- **Outputs:**
  - [To be specified based on {component} requirements]

## Implementation Requirements

### Functional Requirements

- Implement the `{component}()` function in HLS C++ with exactly the same behavior as the reference
- Follow bit-accurate implementation of the reference algorithm (results must match reference within specified error margins)
- Document code thoroughly with comments explaining the algorithm and optimization decisions

### Interface and Data Type Requirements

- Use `hls::stream` interfaces with appropriate buffer depths for streaming data
- Implement fixed-point arithmetic with `ap_fixed<W,I>` (specify exact bit widths based on precision requirements)
- Use `hls::complex<ap_fixed<W,I>>` for any complex number operations
- Define all constant parameters in `{component}.hpp` using `#define` or `constexpr`
- Create descriptive type aliases with `typedef` or `using` statements

### File I/O and Validation only in testbench file `{component}_tb.cpp`

- Read input data from `{component}_in.txt` (one value per line)
- Read threshold values from `threshold_in.txt` (one value per line)
- Read reference output data from `{component}_ref.txt` (one value per line)
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

- Define all constants, types, and function declarations in `{component}.hpp`
- Implement core algorithm in `{component}.cpp` with consistent style
- Follow naming convention: camelCase for variables, PascalCase for types
- Use self-documenting identifier names that clearly reflect their purpose

## Deliverables

- Fully commented HLS C++ implementation files
- Comprehensive testbench demonstrating functional correctness
- Description of optimization approaches and their impact on performance
