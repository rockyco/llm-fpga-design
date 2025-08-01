# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements an LLM-aided FPGA design flow that converts MATLAB algorithms to optimized HLS C++ implementations. The primary focus is on automating the design workflow using Large Language Models (Claude, GPT-4, Gemini) for 5G NR signal processing components, specifically peak picking algorithms for SSB detection.

## Key Architecture Components

### 1. Multi-Stage LLM Pipeline
- **Code Generation**: MATLAB → HLS C++ conversion using structured prompts
- **Automated Debugging**: AI-powered error analysis and code correction
- **Agent Framework**: Orchestrates multiple LLM services with fallback mechanisms

### 2. Core Scripts (scripts/)
- `generate_hls_code.py`: Main code generation from MATLAB to HLS C++
- `debug_assistant.py`: LLM-based debugging of C simulation errors  
- `agent_framework.py`: Agent orchestration and prompt management
- Support for Gemini (primary), OpenAI, and Claude APIs with automatic fallback

### 3. Prompt Engineering System (prompts/)
- Structured templates for different tasks (hls_conversion.md, hls_debugging.md, etc.)
- Domain-specific prompts for 5G signal processing
- Performance optimization and documentation generation templates

## Common Development Commands

### HLS Project Development (in implementations/peakPicker/)
```bash
# Full development cycle
make all                    # Clean, build testbench, run C sim and synthesis

# Individual stages  
make tb                     # Build and run standalone C++ testbench
make csim                   # Run HLS C simulation
make csynth                 # Run HLS C synthesis
make cosim                  # Run C/RTL co-simulation
make export_ip             # Export as IP catalog
make impl                  # Run Vivado implementation
make clean                 # Clean generated files
make help                  # Show all available targets
```

### Code Generation Workflow
```bash
# Generate HLS from MATLAB (from repository root)
python3 scripts/generate_hls_code.py \
  --matlab_file algorithms/peakPicker.m algorithms/peakPicker_tb.m \
  --prompt prompts/hls_conversion.md \
  --model gemini-2.0-flash-thinking-exp

# Debug C simulation errors
python3 scripts/debug_assistant.py \
  --error_log implementations/peakPicker/proj_peakPicker/solution1/csim/report/peakPicker_csim.log \
  --source_file implementations/peakPicker/peakPicker.cpp implementations/peakPicker/peakPicker.hpp implementations/peakPicker/peakPicker_tb.cpp
```

### Environment Setup
```bash
# Required: Set Vitis HLS path
export VITIS_HLS_PATH=/opt/Xilinx/Vitis_HLS/2023.2
source $VITIS_HLS_PATH/settings64.sh

# Required: Set at least one API key
export GEMINI_API_KEY=your_key_here
# Optional alternatives:
export OPENAI_API_KEY=your_key_here  
export CLAUDE_API_KEY=your_key_here

# Install Python dependencies
pip install -r requirements.txt
```

## Hardware Configuration

- **Target FPGA**: xc7k410t-ffg900-2 (Kintex-7)
- **Clock Frequency**: 256MHz (3.9ns period)
- **Clock Uncertainty**: 12.5%
- **HLS Version**: Vitis HLS 2023.2

## LLM Model Selection

### Supported Models and Use Cases
- **gemini-2.0-flash-thinking-exp**: Fast iterations, general debugging
- **gemini-2.0-pro-exp**: Complex algorithm conversion (default)
- **gpt-4**: Detailed implementations requiring careful analysis
- **gpt-3.5-turbo**: Quick prototyping and simple conversions
- **claude-sonnet**: Algorithm explanations and documentation

### API Fallback Order
1. Gemini (primary) - good code reasoning and HLS optimization
2. OpenAI - comprehensive code generation  
3. Claude - detailed algorithmic understanding

## File Organization Patterns

### Implementation Structure
```
implementations/
├── peakPicker/                    # Component-specific directory
│   ├── Makefile                   # HLS build automation
│   ├── peakPicker.cpp/.hpp        # Generated HLS implementation
│   ├── peakPicker_tb.cpp          # Generated testbench
│   └── documentation/             # Auto-generated docs
```

### Data and Testing
- `data/`: Test vectors and reference data (automatically included in HLS builds)
- `algorithms/`: MATLAB reference implementations
- Test data files are automatically detected and added to HLS projects

## Key Implementation Notes

### HLS-Specific Considerations
- All implementations target ap_int<> and ap_fixed<> data types for optimal resource usage
- Testbenches automatically load data from `../../data/` directory
- HLS pragmas are used for performance optimization (PIPELINE, UNROLL, ARRAY_PARTITION)
- Interface synthesis uses ap_ctrl_hs with AXI4-Stream for data

### Code Generation Process
1. Parse MATLAB reference algorithms
2. Apply domain-specific prompts (5G signal processing context)
3. Generate three files: header (.hpp), implementation (.cpp), testbench (_tb.cpp)
4. Automatic verification through C simulation
5. AI-powered debugging if errors occur

### Error Handling and Debugging
- Automatic error log parsing and analysis
- LLM generates detailed debug reports with specific fixes
- Support for interface mismatches, data type issues, and algorithmic errors
- Debug reports saved to `debug_reports/` with timestamps

## Working with Prompts

### Prompt Template Structure
- **Context Section**: Algorithm purpose and background
- **Task Description**: Clear implementation requirements  
- **Technical Requirements**: HLS-specific coding standards
- **Deliverables**: Expected output files and format

### Prompt Backup System
- All prompts are automatically backed up with timestamps in `prompts/backups/`
- Version control for prompt evolution and A/B testing
- Performance metrics tracking for prompt effectiveness