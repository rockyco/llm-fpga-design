# LLM-Aided FPGA Design Flow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HLS Version](https://img.shields.io/badge/HLS-2023.2-blue.svg)](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html)

## Overview

This repository demonstrates a modern approach to FPGA design using Large Language Models (LLMs) to automate and enhance the design workflow from MATLAB algorithms to optimized hardware implementations. By leveraging LLMs like Claude 3.7 Sonnet, GPT-4, or GitHub Copilot, we significantly reduce development time while maintaining design quality.

The repository showcases:

1. Conversion of MATLAB reference algorithms to HLS C++ 
2. Automated debugging of C simulation errors
3. Prompt engineering techniques for hardware design tasks
4. Performance optimization through LLM-guided directives

## Case Study: 5G NR Peak Picker

Our primary example is a peak picker algorithm for 5G NR Synchronization Signal Block (SSB) detection, which demonstrates the complete LLM-assisted workflow from MATLAB specification to optimized HLS implementation.

### Algorithm Description

The peak picker algorithm:
- Takes PSS (Primary Synchronization Signal) correlation magnitude squared values as input
- Compares values against thresholds to identify candidate peaks
- Applies filtering to identify true peaks
- Returns the locations (indices) of detected peaks

## LLM-Based HLS Code Generation and Debugging Workflow

Our comprehensive workflow automates the entire process from MATLAB algorithm to optimized HLS implementation:

```mermaid
graph TB
    subgraph Inputs
        A[MATLAB Prototype Files] -->|Input| B(Generate HLS Code)
        P[Prompt Template] -->|Format| B
    end

    subgraph AI_Code_Generation [AI Code Generation Process]
        B -->|Creates Prompt| C{Select LLM Service}
        C -->|Default| D[Gemini API]
        C -->|Fallback| E[OpenAI API]
        C -->|Fallback| F[Claude API]
        
        D & E & F -->|Generate| G[LLM Response]
        G -->|Parse| H[Extract Code]
        H -->|Save| I[Generated HLS Files]
    end

    subgraph Outputs
        I -->|Header| J[component.hpp]
        I -->|Implementation| K[component.cpp]
        I -->|Testbench| L[component_tb.cpp]
    end

    subgraph Verification
        J & K & L -->|Compile & Run| M[C Simulation]
        M -->|Pass| N[HLS Synthesis]
        M -->|Fail| O[Error Logs]
    end

    subgraph AI_Debug_Assistant [AI Debug Assistant]
        O -->|Input| Q(Debug Assistant)
        J & K & L -->|Source Code| Q
        Q -->|Creates Debug Prompt| R{Select LLM Service}
        R -->|Default| S[Gemini API]
        R -->|Fallback| T[OpenAI API]
        R -->|Fallback| U[Claude API]
        
        S & T & U -->|Analyze| V[LLM Debug Analysis]
        V -->|Generate| W[Debug Report]
        V -->|Extract| X[Code Fixes]
        X -->|Optional| Y[Apply Fixes]
        Y -->|Update| J & K & L
    end

    style D fill:#34A853,stroke:#34A853,color:white
    style S fill:#34A853,stroke:#34A853,color:white
    style G fill:#F9AB00,stroke:#F9AB00,color:white
    style V fill:#F9AB00,stroke:#F9AB00,color:white
    style I fill:#4285F4,stroke:#4285F4,color:white
    style W fill:#4285F4,stroke:#4285F4,color:white
```

### Workflow Stages

#### 1. Input Stage
- **MATLAB Prototype Files**: Reference algorithm implementation in MATLAB
- **Prompt Template**: Structured instructions for the LLM to follow when generating HLS code

#### 2. AI Code Generation Process
- **Creates Prompt**: Combines MATLAB code with template for comprehensive context
- **Select LLM Service**: Chooses between Gemini (default), OpenAI, or Claude APIs
- **LLM Response**: Raw text response containing code and explanations
- **Extract Code**: Parses response to identify different file types and code sections
- **Generated HLS Files**: Creates properly structured C++ files ready for simulation

#### 3. Output Stage
- **Header File**: Contains class definitions, function declarations, and constants
- **Implementation File**: Contains the core HLS algorithm implementation with pragmas
- **Testbench File**: Includes data loading, function calls, and verification logic

#### 4. Verification Stage
- **C Simulation**: Compile and test the generated code for functional correctness
- **HLS Synthesis**: If simulation passes, proceed to hardware synthesis
- **Error Logs**: If simulation fails, collect error information for debugging

#### 5. AI Debug Assistant Stage
- **Debug Assistant**: Takes error logs and source files as input
- **Creates Debug Prompt**: Structures the debugging context for LLM analysis
- **LLM Analysis**: AI analyzes errors and suggests specific code fixes
- **Debug Report**: Comprehensive explanation of issues and solutions
- **Code Fixes**: Specific code changes that can be automatically applied
- **Apply Fixes**: Update source files with AI-suggested corrections

### Prompt Engineering for Code Generation

We've developed specialized prompt templates for effective code generation:

1. **Context Section**: Explains the algorithm purpose and background
2. **Task Description**: Clearly defines what the LLM needs to implement
3. **Implementation Requirements**: Specifies coding standards, interfaces, and optimizations
4. **Deliverables**: Clearly states what files should be produced

Example from our peak picker implementation:

```markdown
# Copilot Instructions for Peak Picker Implementation

## Project Context
This project implements a critical component of a 5G NR SSB detection application. 
The peak picker algorithm identifies SSB signals by locating peaks where the 
magnitude squared of the PSS correlation (`xcorr`) exceeds a predefined threshold.

## Task Description
Your task is to translate the MATLAB peak picker algorithm into efficient HLS C++ 
code while preserving exact functionality. The implementation should be optimized 
for FPGA deployment using Xilinx HLS directives.

[Additional sections...]
```

## How the Debug Assistant Works

The debug assistant provides automated, AI-powered analysis and correction of HLS simulation errors:

```mermaid
graph TD
    subgraph Inputs
        A[Error Log] -->|read_file| C
        B[HLS C++ Source Files] -->|read_file| D
    end

    subgraph Processing
        C[Extract Error Information] --> E
        D[Parse Source Code] --> E
        E[Create Debug Prompt] --> F
    end

    subgraph LLM_Analysis
        F[Query LLM API] -->|model selection| G{Select Model}
        G -->|gemini-2.0-pro-exp| H[Gemini API]
        G -->|gpt-4/gpt-3.5-turbo| I[OpenAI API]
        G -->|claude-sonnet| J[Claude API]
        H --> K[LLM Analysis Response]
        I --> K
        J --> K
    end

    subgraph Outputs
        K --> L[Generate Debug Report]
        K --> M[Parse Code Corrections]
        
        L --> N[Save Markdown Report]
        M --> O[Apply Code Fixes]
        O -->|user confirmation| P[Edit Source Files]
    end

    style H fill:#34A853,stroke:#34A853,color:white
    style K fill:#F9AB00,stroke:#F9AB00,color:white
    style P fill:#4285F4,stroke:#4285F4,color:white
    style N fill:#4285F4,stroke:#4285F4,color:white
```

### Debug Workflow Stages

#### 1. Inputs Processing
- **Error Log Analysis**: Extracts meaningful error patterns from C simulation logs
- **Source Code Parsing**: Gathers relevant source files to provide complete context

#### 2. Processing
- **Extract Error Information**: Identifies specific error messages and patterns
- **Parse Source Code**: Organizes code context for the LLM
- **Create Debug Prompt**: Structures the debugging request with all relevant information

#### 3. LLM Analysis
- **Query LLM API**: Sends the prompt to the selected AI service
- **Model Selection**: Chooses between Gemini (primary), GPT, or Claude models
- **LLM Response**: AI analyzes the issues and provides detailed debugging guidance

#### 4. Outputs
- **Generate Debug Report**: Creates detailed markdown reports explaining errors and fixes
- **Parse Code Corrections**: Extracts specific code changes from the LLM response
- **Apply Code Fixes**: Optionally implements the suggested changes with user confirmation
- **Edit Source Files**: Updates the original files with proper change tracking

The debug assistant handles common HLS errors including:
- Interface mismatches between implementation and testbench
- Data type inconsistencies
- Indexing errors
- Algorithmic logical errors
- Misunderstandings of HLS-specific behaviors

## LLM Selection and Integration

Our tools support multiple LLM providers with different capabilities:

- **Gemini Pro/Flash**: Offers strong reasoning about code structures and efficient debugging
- **GPT-3.5/4**: Provides detailed code generation with comprehensive comments
- **Claude Sonnet**: Excels at understanding complex algorithms and providing thorough explanations

The framework automatically selects appropriate models based on task complexity, or allows specifying a model for specific use cases.

## Automated File Generation and Management

The `generate_hls_code.py` tool implements sophisticated code extraction algorithms to:

- Parse LLM responses for code blocks
- Identify appropriate file types (header, implementation, testbench)
- Generate properly formatted HLS C++ files
- Maintain correct dependencies between files
- Create project structures compatible with Vitis HLS

## Getting Started

### Prerequisites

- Vitis HLS 2023.2 or newer
- MATLAB R2023a or newer (for reference models)
- Python 3.8+ with necessary libraries for data handling
- API keys for supported LLM services (at least one of the following):
  - Google Gemini API key (recommended)
  - OpenAI API key
  - Anthropic Claude API key

### Installation

```bash
# Clone this repository
git clone https://github.com/rockyco/llm-fpga-design.git
cd llm-fpga-design

# Set up your environment
source /path/to/Vitis/settings64.sh

# Install required Python packages
pip install -r requirements.txt

# Add your API keys to the .bashrc or .env file
echo "GEMINI_API_KEY=your_gemini_api_key" >> ~/.bashrc
echo "OPENAI_API_KEY=your_openai_api_key" >> ~/.bashrc
echo "CLAUDE_API_KEY=your_claude_api_key" >> ~/.bashrc
source ~/.bashrc
```

### Usage

1. **Generate HLS C++ from MATLAB reference**:
   Supported models: `gemini-2.0-flash-thinking-exp`, `gemini-2.0-pro-exp`, `gpt-4`, `gpt-3.5-turbo`, `claude-sonnet`
   ```bash
   python3 scripts/generate_hls_code.py --matlab_file algorithms/peakPicker.m algorithms/peakPicker_tb.m --prompt prompts/hls_conversion.md --model gemini-2.0-flash-thinking-exp
   ```

2. **Run C simulation**:
   ```bash
   cd implementations/peakPicker
   make csim
   ```

3. **Debug errors with LLM assistance**:
   ```bash
   cd ../../
   python3 scripts/debug_assistant.py --error_log implementations/peakPicker/proj_peakPicker/solution1/csim/report/peakPicker_csim.log --source_file implementations/peakPicker/peakPicker.cpp implementations/peakPicker/peakPicker.hpp implementations/peakPicker/peakPicker_tb.cpp
   ```

4. **Synthesize and export RTL**:
   ```bash
   make csynth
   make export_ip
   ```

## Code Generation Process

The `generate_hls_code.py` script implements a comprehensive code generation pipeline:

1. **Code Analysis**: Examines MATLAB reference to understand algorithm function
2. **Prompt Construction**: Combines specialized templates with example code
3. **Model Selection**: Uses the most appropriate LLM based on task needs
4. **Response Processing**: Implements robust parsing to extract code blocks
5. **Code Organization**: Creates properly structured HLS project files
6. **Documentation**: Automatically preserves explanations from the LLM

Key features include:
- Support for multi-file MATLAB input
- Robust code block extraction with multiple fallback strategies
- File type identification based on content patterns
- Project structure generation following HLS best practices

## Repository Structure

```
llm-fpga-design/
├── algorithms/                  # MATLAB reference implementations
├── implementations/             # Generated HLS C++ implementations
│   └── peakPicker/              # Peak Picker implementation case study
├── prompts/                     # LLM prompt templates
├── scripts/                     # Automation scripts
│   ├── generate_hls_code.py     # Code generation script
│   └── debug_assistant.py       # Debugging assistant script
├── data/                        # Test data files
└── docs/                        # Documentation
```

## Best Practices

Based on our experience, we recommend these best practices for LLM-assisted FPGA design:

1. **Structured Prompts**: Use clear, detailed prompts with specific sections for context, requirements, and deliverables
2. **Iterative Refinement**: Start with high-level requirements, then refine implementation details
3. **Input/Output Examples**: Provide concrete examples of expected behavior
4. **Domain-Specific Knowledge**: Include relevant HLS and FPGA concepts in prompts
5. **Error Analysis**: When debugging, provide complete error messages and surrounding context
6. **Model Selection**: Choose appropriate models for different tasks:
   - Use Gemini Flash for quick iterations and debugging
   - Use GPT-4 for complex algorithms needing careful implementation
   - Use Claude for detailed explanations and educational contexts
7. **Prompt Templates**: Maintain a library of effective prompt templates for reuse
8. **Human Review**: Always review and understand generated code before synthesis

## Limitations and Considerations

- LLMs may not be aware of the latest HLS features or hardware-specific optimizations
- Complex timing constraints might require manual refinement
- While LLMs can generate optimized code, expert review is still recommended for critical applications
- Actual hardware performance should be verified through physical implementation
- LLMs may occasionally:
  - Generate incorrect pragma syntax that needs manual correction
  - Not fully understand resource vs. performance tradeoffs
  - Struggle with very complex interface requirements
  - Need help with target-specific optimizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the open-source HLS and FPGA design communities
- Special thanks to the developers of Google Gemini 2.5 pro API, Claude 3.7 Sonnet, and GitHub Copilot for enabling this workflow

