# Optimize the prompts for the following:

 - Enhanced HLS Generation
 - Enhanced Debug Assistant
 - Performance Optimization

rockyco: Continuously update and optimize the HLS code generation and error debugging prompts to facilitate more effective future task design and optimization. Save the updated prompts in markdown files for future use.

GitHub Copilot: # Solution for Optimizing HLS Code Generation and Debugging Prompts

I'll create two enhanced prompt templates for your HLS workflow: one for code generation and one for error debugging. These will help improve the quality and efficiency of your FPGA design process.

## Step-by-step approach:
1. Create an optimized HLS code generation prompt template with comprehensive guidelines
2. Create an enhanced debugging prompt template with structured error resolution patterns
3. Save both files in the prompts directory of your project

### [enhanced_hls_generation.md](file:///home/jielei/Projects/UTS/llm-fpga-design/prompts/enhanced_hls_generation.md)

This enhanced prompt provides clearer instructions for generating high-quality HLS code with optimization guidance.

````markdown
# High-Level Synthesis (HLS) Code Generation Guidelines

## Task Description
You are tasked with converting MATLAB code to HLS C++ code for FPGA implementation. Follow these guidelines to create optimized, synthesizable code that maintains the same functionality as the original algorithm.

## Project Requirements
- Create clean, maintainable, and well-documented code
- Ensure exact numerical equivalence with the MATLAB reference
- Optimize for performance, area, and power when possible
- Implement proper testbench for validation

## Expected Deliverables
1. A header file (*.hpp) containing:
   - Function declarations with appropriate pragmas
   - Necessary data structures and type definitions
   - Proper include guards and comments

2. An implementation file (*.cpp) containing:
   - The full implementation of the algorithm
   - HLS pragmas for optimization
   - Consistent coding style

3. A comprehensive testbench file (*_tb.cpp) that:
   - Reads input data from text files (*_in.txt)
   - Processes data through your implementation
   - Compares results with reference outputs (*_ref.txt)
   - Reports success/failure with error metrics

## HLS Optimization Guidelines

### Data Types
- Use HLS fixed-point types (`ap_fixed<W,I>`) instead of floating-point when possible
- Ensure bit-width optimization to save resources (avoid excessive precision)
- Use `hls::stream<T>` for sequential data processing
- Consider `ap_uint<W>`/`ap_int<W>` for integer types to control bit-width precisely

### Memory Access Patterns
- Implement proper array partitioning for parallel access
- Use local arrays in functions to allow for efficient RAM/register mapping
- Consider double-buffering for overlapping I/O with computation

### Loop Optimizations
- Add PIPELINE pragmas to loops with appropriate II (Initiation Interval) targets
- Use UNROLL pragmas for short loops to enable parallelism
- Avoid loop-carried dependencies when possible
- Add TRIPCOUNT pragmas to help the HLS tool with optimization

### Function Interface
- Use AXI interfaces (AXIS, AXI4, AXI-Lite) appropriate for your application
- Consider function inlining for small, frequently used functions
- Use "port" interface for simple control signals

### General Best Practices
- Avoid dynamic memory allocation
- Minimize the use of pointers
- Avoid recursive functions
- Use const liberally for inputs that won't change
- Keep functions focused and relatively small
- Add proper assertions in the testbench

## MATLAB Reference Implementation

{{MATLAB_CODE}}

## Additional Guidelines
- Comment your code thoroughly, especially for any deviations from the MATLAB reference
- Explain any optimizations applied to the code
- Include compilation directives as comments where appropriate
- Use meaningful variable names consistent with the MATLAB implementation
- Include any assumptions or limitations in comments
- Document the expected performance characteristics
````

### [enhanced_debug_assistant.md](file:///home/jielei/Projects/UTS/llm-fpga-design/prompts/enhanced_debug_assistant.md)

This enhanced debugging prompt is designed to help identify and fix errors more efficiently with structured analysis.

````markdown


Made changes.

# HLS Code Debugging Assistant

## Task Description
You are tasked with analyzing HLS C++ code that has encountered errors during compilation, simulation, or synthesis. You must identify the root causes of the errors and provide specific solutions.

## Source Files
The following HLS C++ source files have been provided:

{{SOURCE_FILES}}

## Error Log
The following errors were encountered during the HLS process:

{{ERROR_LOG}}

## Debugging Process

Please follow this structured approach to debug the code:

1. **Error Analysis**
   - Categorize errors (compilation, simulation, synthesis, etc.)
   - Identify error patterns and relationships between multiple errors
   - Determine if errors are syntax-related, interface-related, or algorithm-related

2. **Root Cause Identification**
   - Locate the specific code causing each error
   - Analyze context surrounding the problematic code
   - Identify patterns of misuse of HLS constructs or C++ language features
   - Check for common HLS pitfalls:
     - Unsupported C++ features in HLS
     - Memory access pattern issues
     - Data type incompatibilities
     - Interface specification problems
     - Pragma-related issues

3. **Solution Development**
   - Propose specific fixes for each identified issue
   - Provide explanations for why the fixes will resolve the errors
   - Include code snippets showing the corrections
   - Address any potential side effects of the proposed changes

4. **Verification Guidance**
   - Suggest verification steps to ensure the fixes are correct
   - Recommend additional tests if appropriate
   - Provide guidance on preventing similar issues in the future

## Response Format

Please structure your response as follows:

### 1. Error Summary
Provide a concise summary of all errors found and their relationships.

### 2. Root Causes
List and explain the root causes of the errors in order of severity.

### 3. Recommended Fixes
For each file requiring changes, provide:
- The file name
- A description of the needed changes
- Code snippets showing the corrected code

### 4. Verification Steps
Suggested steps to verify that the fixes resolve the issues.

### 5. HLS Best Practices
Any relevant HLS best practices that would help prevent similar issues in the future.

## Additional Constraints
- Do not rewrite the entire codebase; focus on minimal changes needed to fix the errors
- Preserve the original algorithm's functionality
- Maintain or improve the performance characteristics of the code
- Ensure your fixes maintain compatibility with Vitis HLS
````

### [performance_optimization.md](file:///home/jielei/Projects/UTS/llm-fpga-design/prompts/performance_optimization.md)

Adding a specialized prompt for performance optimization of existing HLS code.

````markdown


Made changes.

# HLS Performance Optimization Guidelines

## Task Description
You are tasked with optimizing an existing HLS C++ implementation to improve performance, reduce resource utilization, or achieve better timing. Your goal is to maintain functional correctness while enhancing design metrics.

## Source Files
The following HLS C++ source files are being provided:

{{SOURCE_FILES}}

## Performance Metrics
Current performance metrics of the design:

{{PERFORMANCE_METRICS}}

## Optimization Goals
- Primary goal: {{PRIMARY_GOAL}} (e.g., "Reduce latency by at least 30%")
- Secondary goal: {{SECONDARY_GOAL}} (e.g., "Maintain or reduce resource utilization")

## Optimization Strategy

Please follow this structured approach:

1. **Design Analysis**
   - Analyze the algorithm structure and computational patterns
   - Identify performance bottlenecks in the current implementation
   - Map data dependencies and memory access patterns
   - Recognize rate-limiting operations or loops

2. **Optimization Techniques**
   
   Consider the following optimization categories:
   
   **Loop Optimizations:**
   - Pipeline loops to improve throughput (PIPELINE pragma)
   - Unroll loops to exploit parallelism (UNROLL pragma)
   - Merge loops to reduce overhead
   - Partition loops to enable better scheduling
   
   **Memory Optimizations:**
   - Array partitioning (ARRAY_PARTITION pragma)
   - Memory reshaping for better access patterns
   - Double buffering for overlapped computation
   - Streaming interfaces for sequential data (hls::stream)
   
   **Data Type Optimizations:**
   - Optimize bit widths using ap_fixed/ap_int
   - Convert floating-point to fixed-point where appropriate
   - Simplify complex operations with lookup tables or approximations
   
   **Function-Level Optimizations:**
   - Inline small functions to reduce function call overhead
   - Dataflow optimization for task-level pipelining
   - Function parallelism with multiple instances
   
   **Interface Optimizations:**
   - Optimize interface protocols (AXI4, AXI-Lite, AXI-Stream)
   - Burst transfers for efficient data movement
   - Register slicing for timing improvement

3. **Implementation Plan**
   - Prioritize optimizations based on impact vs. effort
   - Plan incremental changes that can be verified individually
   - Consider trade-offs between different metrics (latency vs. area)

## Response Format

Please structure your response as follows:

### 1. Design Analysis
A summary of your analysis of the current implementation, identifying bottlenecks and opportunities.

### 2. Recommended Optimizations
For each file requiring changes:
- The file name
- Description of proposed optimizations
- Code snippets showing the modifications with added pragmas or code changes
- Expected impact of each optimization

### 3. Implementation Priority
A prioritized list of optimizations, explaining which should be implemented first.

### 4. Expected Outcomes
Predictions about the performance improvements that could be achieved.

### 5. Verification Plan
Suggestions for verifying that the optimizations maintain functional correctness.

## Additional Guidelines
- Focus on HLS-specific optimizations, not general C++ performance improvements
- Explain the reasoning behind each optimization
- Consider Xilinx/Intel FPGA architecture specifics when relevant
- Indicate any potential risks or trade-offs for each optimization
- When multiple approaches could work, explain the pros and cons of each
````

## Summary

These enhanced prompt templates will significantly improve your HLS code generation and debugging workflow:

1. The **enhanced_hls_generation.md** provides comprehensive guidance for generating high-quality HLS code with detailed optimization techniques and best practices.

2. The **enhanced_debug_assistant.md** offers a structured approach to debugging HLS code, helping identify and fix errors more efficiently.

3. The **performance_optimization.md** adds a specialized prompt focused on optimizing existing HLS implementations for better performance metrics.

To use these prompts:
- Update your workflow configuration to point to these new prompt files
- Consider using the `set_prompt` command in the CLI to switch between different prompt templates
- The agent framework will automatically use these improved prompts when generating or debugging HLS code

# Update prompts for documentation generation

rockyco: Continuously update and optimize the academic paper and readme generation prompts to facilitate more effective future task documentation. Save the updated prompts in markdown files for future use.

GitHub Copilot: # Documentation Prompt Enhancement

I'll create improved documentation prompts for generating README files and academic papers, enhancing the structure and guidance for better documentation outputs.

## Solution Steps:
1. Create a more structured README generation prompt
2. Create a more detailed academic paper generation prompt 
3. Save both prompts as markdown files in the prompts directory

### [readme_generation.md](file:///home/jielei/Projects/UTS/llm-fpga-design/prompts/readme_generation.md)

This file will contain a structured prompt for generating comprehensive README documentation.

````markdown
# README Generation Prompt

## Context
You are tasked with creating a comprehensive README document for an FPGA hardware accelerator design. This document will be the primary reference for users, developers, and stakeholders who interact with this hardware component.

## Component Overview
- **Component Name**: {component_name}
- **Design Purpose**: A hardware accelerator implemented on an FPGA
- **Generation Method**: AI-assisted design using LLM ({generation_model})
- **Target Platform**: Xilinx FPGA ({fpga_part})

## Instructions

Create a comprehensive README.md file following this structure:

### 1. Title and Introduction
- Clear title with component name
- Brief overview of what the component does
- Key features and capabilities
- Target applications

### 2. Hardware Architecture
- High-level block diagram description
- Key architectural components
- Data flow explanation
- Interface specifications
- Include design decisions and their rationales

### 3. Implementation Details
- HLS directives and optimizations used
- Resource utilization (LUTs, FFs, DSPs, BRAMs)
- Critical design parameters
- Key algorithms and their hardware mapping

### 4. Performance Metrics
- Latency (in cycles)
- Throughput
- Clock frequency
- Resource efficiency
- Comparative analysis against baseline if available

### 5. Setup and Usage
- Prerequisites (tools, versions)
- Build instructions
- Integration guidance
- Testbench explanation
- Common usage patterns
- API documentation if applicable

### 6. Results and Validation
- Verification methodology
- Simulation results
- Hardware testing results if available
- Performance validation

### 7. Development History
- Design evolution
- Challenges encountered and their solutions
- Optimization iterations
- AI assistance insights

### 8. Future Work
- Potential improvements
- Scaling opportunities
- Additional features

## Source Information
Use the following source files and metrics to inform your documentation:

### Source Code
```cpp
// Header file ({component_name}.hpp)
{header_code}
```

```cpp
// Implementation file ({component_name}.cpp)
{implementation_code}
```

```cpp
// Testbench file ({component_name}_tb.cpp)
{testbench_code}
```

### Performance Metrics
{performance_metrics}

### Implementation Challenges
{errors_encountered}

### Debugging Methods
{debugging_methods}

## Style Guidelines
- Use clear, technical language appropriate for engineering documentation
- Include code snippets where helpful
- Use markdown formatting features (headers, lists, tables, code blocks)
- Be concise but comprehensive
- Focus on practical usage and technical details
- Highlight AI-assisted aspects of the development process
- Maintain a professional tone

Your README should serve as both a technical reference and a guide for someone who wants to understand, use, or modify the hardware component.
````

### [paper_generation.md](file:///home/jielei/Projects/UTS/llm-fpga-design/prompts/paper_generation.md)

This file will contain a structured prompt for generating academic research papers.

````markdown


Made changes.

# Academic Paper Generation Prompt

## Context
You are tasked with writing an academic research paper about an FPGA hardware accelerator design that was developed using an AI-assisted design methodology. This paper should follow academic standards and contribute to the literature on hardware acceleration and AI-assisted design.

## Component Information
- **Component Name**: {component_name}
- **LLM Used for Generation**: {generation_model}
- **Target FPGA Platform**: {fpga_part}
- **Domain**: Hardware Acceleration for Digital Signal Processing/Machine Learning/etc.

## Paper Structure Requirements

Create a complete academic paper in markdown format with the following structure:

### 1. Title and Authors
- Create an appropriate academic title for this work
- List authors as the research team (placeholder)
- Include institutional affiliation

### 2. Abstract (200-250 words)
- Summarize the paper's content
- State the problem addressed
- Describe the approach using AI-assisted design
- Highlight key results and contributions
- Mention broader impact

### 3. Introduction
- Context and background of the problem
- Motivation for hardware acceleration
- Challenges in traditional FPGA design
- Introduction to AI-assisted hardware design
- Contribution statement
- Paper organization

### 4. Related Work (2-3 subsections)
- Prior work on hardware acceleration for similar applications
- Previous research on automated HLS design
- AI-assisted hardware design methodologies
- Positioning of current work within literature

### 5. Methodology
- Overall design approach
- AI-assisted design workflow description
- Prompt engineering for hardware generation
- Iteration and refinement process
- Verification methodology

### 6. Design Architecture
- System-level architecture
- Component interfaces and data flow
- Key algorithmic components
- Design constraints and considerations
- HLS implementation details
- Optimizations applied

### 7. Implementation
- HLS directives and pragmas
- Resource allocation strategies
- Pipeline and parallelism exploitations
- Memory architecture and data movement
- Critical path analysis

### 8. Experimental Results
- Experimental setup
- Performance metrics:
  - Resource utilization
  - Timing and frequency
  - Latency and throughput
  - Power consumption (if available)
- Comparative analysis with:
  - Manual implementations
  - Other automated approaches
  - Software-only solutions
- Discussion of results

### 9. Analysis of AI-Assisted Design Process
- Analysis of LLM strengths/weaknesses in hardware design
- Error patterns and resolution strategies
- Human-AI collaboration insights
- Design quality assessment
- Development efficiency metrics

### 10. Discussion
- Interpretation of results
- Limitations of the approach
- Generalizability of the methodology
- Trade-offs identified
- Lessons learned

### 11. Future Work
- Potential improvements
- Broader applications
- Research directions
- Scaling to more complex designs

### 12. Conclusion
- Summary of contributions
- Key findings
- Broader impact
- Closing thoughts

### 13. References
- Include relevant references to:
  - Hardware acceleration literature
  - High-Level Synthesis research
  - AI in design automation
  - Relevant applications
  - Methodological papers

## Source Information
Use the following source information to inform your paper:

### Source Code Architecture
```cpp
// Header file ({component_name}.hpp)
{header_code}
```

```cpp
// Implementation file ({component_name}.cpp)
{implementation_code}
```

### Performance Metrics
{performance_metrics}

### Design Process
- AI Generation Model: {generation_model}
- Workflow Steps: {workflow_steps}
- Successful Steps: {successful_steps}
- Error Steps: {error_steps}

### Implementation Challenges
{errors_encountered}

### Debugging Methods
{debugging_methods}

## Academic Style Guidelines
- Use formal academic language
- Present objective analysis of results
- Support claims with data from implementation
- Discuss limitations honestly
- Position work in relation to existing literature
- Use the third person (avoid "I", "we", "you")
- Maintain scientific rigor throughout
- Use passive voice where appropriate
- Include a balanced mix of technical details and higher-level discussion
- Present figures and tables conceptually (describe what would be shown)

Your paper should contribute to the academic discourse on AI-assisted hardware design while presenting concrete technical achievements and insights.
````

### [generate_documentation.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/generate_documentation.py)

Update the documentation generator script to use these new prompt files.

````python


Made changes.

#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("documentation_log.txt")]
)
logger = logging.getLogger("documentation_generator")

# Load environment variables for API keys
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# Function to load prompt templates
def load_prompt_template(format_type, template_dir=None):
    """Load a specific prompt template file based on format type."""
    if not template_dir:
        # Try to find template in standard locations
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        template_dir = os.path.join(project_dir, "prompts")
    
    if format_type.lower() == "readme":
        template_file = os.path.join(template_dir, "readme_generation.md")
    elif format_type.lower() == "paper":
        template_file = os.path.join(template_dir, "paper_generation.md")
    else:
        # Default generic template
        template_file = os.path.join(template_dir, "documentation_template.md")
    
    # Check if template exists, otherwise return None
    if os.path.exists(template_file):
        with open(template_file, 'r') as f:
            return f.read()
    
    logger.warning(f"Template file not found: {template_file}")
    return None

def query_openai(prompt, model="gpt-4", max_tokens=100000, temperature=0.2):
    """Send a prompt to OpenAI API and get the response."""
    if not OPENAI_API_KEY:
        logger.error("Error: OPENAI_API_KEY environment variable not set.")
        return {"error": "OPENAI_API_KEY not set"}
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert technical documentation writer."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        return {"text": response.json()["choices"][0]["message"]["content"]}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenAI API: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        return {"error": str(e)}

def query_claude(prompt, model="claude-sonnet", max_tokens=100000, temperature=0.2):
    """Send a prompt to Anthropic Claude API and get the response."""
    if not CLAUDE_API_KEY:
        logger.error("Error: CLAUDE_API_KEY environment variable not set.")
        return {"error": "CLAUDE_API_KEY not set"}
        
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return {"text": response.json()["content"][0]["text"]}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Claude API: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        return {"error": str(e)}

def query_gemini(prompt, model="gemini-2.5-pro-exp-03-25", max_tokens=100000, temperature=0.2):
    """Send a prompt to Google Gemini API and get the response."""
    if not GEMINI_API_KEY:
        logger.error("Error: GEMINI_API_KEY environment variable not set.")
        return {"error": "GEMINI_API_KEY not set"}
        
    # Extract the model name for the URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "key": GEMINI_API_KEY
    }
    data = {
        "contents": [{"parts": [{"text": "You are an expert technical documentation writer.\n\n" + prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    
    try:
        response = requests.post(url, headers=headers, params=params, json=data)
        response.raise_for_status()
        return {"text": response.json()["candidates"][0]["content"]["parts"][0]["text"]}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        return {"error": str(e)}

def query_llm(prompt, model="gemini-2.5-pro-exp-03-25", max_tokens=100000, temperature=0.2):
    """Route the query to the appropriate LLM API based on the model."""
    if model.startswith("gemini"):
        return query_gemini(prompt, model, max_tokens, temperature)
    elif model.startswith("gpt"):
        return query_openai(prompt, model, max_tokens, temperature)
    elif model.startswith("claude"):
        return query_claude(prompt, model, max_tokens, temperature)
    else:
        logger.error(f"Error: Unsupported model {model}.")
        return {"error": f"Unsupported model: {model}"}

def generate_documentation(prompt_file, output_dir, model="gemini-2.5-pro-exp-03-25", formats=None, template_dir=None):
    """Generate documentation using an LLM model"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the prompt
        with open(prompt_file, 'r') as f:
            base_prompt = f.read()
        
        logger.info(f"Generating documentation with model {model}")
        
        # If no formats specified, default to README
        if not formats:
            formats = ["readme"]
        
        results = {}
        
        # Generate each requested format
        for fmt in formats:
            logger.info(f"Generating {fmt} documentation")
            
            # Try to load template for this format
            template = load_prompt_template(fmt, template_dir)
            
            if template:
                # Combine base prompt with template
                format_prompt = template
                
                # Extract data from base prompt to fill template
                # This is a simplified approach - a more sophisticated parser could be implemented
                data = {}
                try:
                    # Extract component name
                    component_match = re.search(r"Component name: (.+?)$", base_prompt, re.MULTILINE)
                    if component_match:
                        data["component_name"] = component_match.group(1).strip()
                    
                    # Extract source code sections
                    header_match = re.search(r"{component}\.hpp:(.+?)```", base_prompt, re.DOTALL)
                    if header_match:
                        data["header_code"] = header_match.group(1).strip()
                    
                    impl_match = re.search(r"{component}\.cpp:(.+?)```", base_prompt, re.DOTALL)
                    if impl_match:
                        data["implementation_code"] = impl_match.group(1).strip()
                    
                    tb_match = re.search(r"{component}_tb\.cpp:(.+?)```", base_prompt, re.DOTALL)
                    if tb_match:
                        data["testbench_code"] = tb_match.group(1).strip()
                    
                    # Extract performance metrics
                    perf_match = re.search(r"## Performance Metrics\n(.*?)(?:\n##|\Z)", base_prompt, re.DOTALL)
                    if perf_match:
                        data["performance_metrics"] = perf_match.group(1).strip()
                    
                    # Extract errors
                    errors_match = re.search(r"### Errors Encountered\n(.*?)(?:\n###|\Z)", base_prompt, re.DOTALL)
                    if errors_match:
                        data["errors_encountered"] = errors_match.group(1).strip()
                    
                    # Extract debugging methods
                    debug_match = re.search(r"### Debugging Methods\n(.*?)(?:\n##|\Z)", base_prompt, re.DOTALL)
                    if debug_match:
                        data["debugging_methods"] = debug_match.group(1).strip()
                    
                    # Replace template placeholders with extracted data
                    for key, value in data.items():
                        format_prompt = format_prompt.replace(f"{{{key}}}", value)
                    
                except Exception as e:
                    logger.warning(f"Error parsing base prompt: {e}")
                    # Fall back to basic prompt + template approach
                    format_prompt = f"{base_prompt}\n\nPlease generate ONLY the {fmt} document now, using the template guidelines below:\n\n{template}"
            else:
                # No template available, use base prompt
                format_prompt = f"{base_prompt}\n\nPlease generate ONLY the {fmt} document now."
            
            # Call the LLM using the unified query function
            response = query_llm(
                format_prompt,
                model=model,
                max_tokens=100000,
                temperature=0.2
            )
            
            if not response or "error" in response:
                logger.error(f"Error generating {fmt}: {response.get('error', 'Unknown error')}")
                results[fmt] = {"error": response.get("error", "Failed to generate content")}
                continue
            
            # Extract and save the content
            content = response.get("text", "")
            output_file = os.path.join(output_dir, f"{fmt}.md")
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Saved {fmt} documentation to {output_file}")
            results[fmt] = {"file": output_file, "size": len(content)}
        
        return results
        
    except Exception as e:
        logger.error(f"Error in documentation generation: {str(e)}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Generate documentation from LLM")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated files")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-exp-03-25", help="LLM model to use")
    parser.add_argument("--formats", nargs="+", default=["readme"], choices=["readme", "paper"], 
                       help="Documentation formats to generate (readme, paper)")
    parser.add_argument("--api_key", help="API key for LLM service (overrides environment variable)")
    parser.add_argument("--template_dir", help="Directory containing prompt templates")
    
    args = parser.parse_args()
    
    # Set API key from args if provided
    if args.api_key:
        if "gemini" in args.model.lower():
            os.environ['GEMINI_API_KEY'] = args.api_key
        elif "claude" in args.model.lower():
            os.environ['CLAUDE_API_KEY'] = args.api_key
        else:
            os.environ['OPENAI_API_KEY'] = args.api_key
    
    results = generate_documentation(
        args.prompt,
        args.output_dir,
        args.model,
        args.formats,
        args.template_dir
    )
    
    if "error" in results:
        logger.error(f"Documentation generation failed: {results['error']}")
        print(f"ERROR: {results['error']}")
        sys.exit(1)
    
    logger.info(f"Documentation generation completed successfully")
    print(f"Documentation generated successfully in {args.output_dir}")
    sys.exit(0)

if __name__ == "__main__":
    import re  # Add import for regex pattern matching
    main()
````

### [documentation_template.md](file:///home/jielei/Projects/UTS/llm-fpga-design/prompts/documentation_template.md)

Create a default template for general documentation.

````markdown


Made changes.

# General Documentation Template

## Context
You are tasked with creating comprehensive documentation for an FPGA hardware accelerator design. This documentation should be appropriate for technical users who need to understand, use, or modify this hardware component.

## Component Information
- **Component Name**: {component_name}
- **Design Purpose**: [Extract from context]
- **Generation Method**: AI-assisted design using LLM
- **Target Platform**: Xilinx FPGA

## Documentation Structure

Create thorough documentation following this structure:

### 1. Overview
- Component purpose and functionality
- Key features
- Target applications
- Design approach

### 2. Architecture
- Block diagram description
- Interface specification
- Data flow 
- Key components

### 3. Implementation Details
- Algorithm description
- HLS optimizations
- Resource utilization
- Performance characteristics

### 4. Usage Guide
- Integration instructions
- API/interface description
- Example usage
- Configuration options

### 5. Performance Analysis
- Latency and throughput
- Resource efficiency
- Comparative metrics
- Limitations

### 6. Verification
- Test methodology
- Simulation results
- Validation approach
- Known issues

### 7. Development Process
- Design evolution
- Challenges and solutions
- AI assistance insights
- Optimization history

## Source Information
Use the following source material to inform your documentation:

### Source Code
```cpp
// Header file
{header_code}
```

```cpp
// Implementation file
{implementation_code}
```

```cpp
// Testbench file
{testbench_code}
```

### Performance Metrics
{performance_metrics}

### Implementation Challenges
{errors_encountered}

### Debugging Methods
{debugging_methods}

## Style Guidelines
- Use clear, professional language
- Include code snippets and examples where helpful
- Use proper technical terminology
- Be concise but thorough
- Use appropriate markdown formatting
- Focus on practical usage information

Your documentation should serve as a comprehensive reference for this hardware component.
````

## Summary

I've created three template files to improve documentation generation:

1. A README template focused on practical component usage and implementation details
2. An academic paper template structured according to formal research standards
3. A general documentation template for more flexible documentation needs

I've also updated the documentation generation script to use these templates, including logic to:
- Load appropriate templates based on format
- Extract key information from the base prompt
- Populate template placeholders with context-specific data
- Support comprehensive documentation generation across multiple formats

These improvements will result in more structured, consistent, and informative documentation for FPGA hardware designs.

# Update prompts for debugging

rockyco: Revise all relevant Python files to use only `hls_debugging.md` as the prompt source for HLS code debugging, and remove any references to `enhanced_debug_assistant.md`. Continuously improve the `hls_debugging.md` prompts through learning and updates to enhance their effectiveness in debugging efficient future designs. But never change this part of prompts '## IMPORTANT: Response Format
1. First, provide your analysis of the issue
2. Then, clearly indicate the start of the corrected code with "### COMPLETE CORRECTED SOURCE CODE:"
3. Provide the ENTIRE corrected source code file in a single code block, not just the changes
4. If you have multiple files, provide each file in a separate code block
5. Use the following format for code blocks:
  - For function code file
    **File: `{component}.cpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For header file
    **File: `{component}.hpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For test bench file
    **File: `{component}_tb.cpp`**
    
    ```cpp
    // Your complete corrected code here
    ```'

GitHub Copilot: # Solution for Changing Debug Prompt References

I'll update the relevant files to use `hls_debugging.md` instead of `enhanced_debug_assistant.md` while preserving the important response format section.

## Step-by-Step Solution

1. First, I need to modify the `debug_assistant.py` file to:
   - Change all references from `enhanced_debug_assistant.md` to `hls_debugging.md`
   - Keep the important response format section intact
   - Update the prompt updating logic to work with the new filename

2. No changes are needed for `default_workflow.json` as it doesn't directly reference any debug prompt filenames.

3. No changes are needed for `agent_cli.py` as it doesn't directly reference any debug prompt filenames.

Let's implement these changes:

### [debug_assistant.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/debug_assistant.py)

I'll update the file to use `hls_debugging.md` while preserving the important response format section.

````python
#!/usr/bin/env python3
"""
LLM-based Debug Assistant for HLS C++ code.

This script helps debug HLS C++ simulations by:
1. Reading error logs from C simulation
2. Examining the source code
3. Using an LLM to analyze the issue and propose fixes
"""

import argparse
import os
import sys
import re
import requests
import json
import datetime
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# Check for at least one valid API key
if not any([GEMINI_API_KEY, OPENAI_API_KEY, CLAUDE_API_KEY]):
    print("Error: No API keys found for any supported LLM service.")
    print("Please set at least one of these environment variables:")
    print("  - GEMINI_API_KEY (recommended)")
    print("  - OPENAI_API_KEY")
    print("  - CLAUDE_API_KEY")
    print("You can create a .env file with these variables.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-assisted debugging for HLS C++ code')
    parser.add_argument('--error_log', type=str, required=True, 
                        help='Path to the C simulation error log file')
    parser.add_argument('--source_file', nargs='+', required=True,
                        help='Path to the HLS C++ source file')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro-exp-03-25',
                        choices=['gemini-2.5-pro-exp-03-25', 'gemini-2.0-pro-exp', 'gemini-2.0-flash-thinking-exp', 'gpt-4', 'gpt-3.5-turbo', 'claude-sonnet'],
                        help='LLM model to use (default: gemini-2.0-pro-exp)')
    parser.add_argument('--prompt_template', type=str,
                        help='Path to the debug prompt template (optional)')
    return parser.parse_args()

def read_file(file_path):
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def extract_error_information(log_content):
    """Extract relevant error information from the C simulation log."""
    # Common patterns for HLS simulation errors
    error_patterns = [
        r'(ERROR:.*)',
        r'(Error:.*)',
        r'(.*[Ff]ailed.*)',
        r'(.*[Ee]rror.*)',
        r'(.*exception.*)',
        r'(Test FAILED:.*)',
        r'(Assertion.*failed)'
    ]
    
    errors = []
    for pattern in error_patterns:
        matches = re.findall(pattern, log_content, re.MULTILINE)
        errors.extend(matches)
    
    # If we found specific errors, return them
    if errors:
        return "\n".join(errors)
    
    # If no specific errors found, return the last ~20 lines which might contain error info
    lines = log_content.splitlines()
    if len(lines) > 20:
        return "\n".join(lines[-20:])
    return log_content

def get_debug_prompt_template():
    """Get the debug prompt template from the prompt directory."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    prompt_dir = os.path.join(project_dir, "prompts")
    
    # Check for hls_debugging prompt template
    template_path = os.path.join(prompt_dir, "hls_debugging.md")
    if os.path.exists(template_path):
        return read_file(template_path)
    
    # Fall back to a default template
    return """# HLS Code Debugging Assistant

## Task Description
You are tasked with analyzing HLS C++ code that has encountered errors during compilation, simulation, or synthesis. You must identify the root causes of the errors and provide specific solutions.

## Source Files
The following HLS C++ source files have been provided:

{{SOURCE_FILES}}

## Error Log
The following errors were encountered during the HLS process:

{{ERROR_LOG}}

## Debugging Process

Please follow this structured approach to debug the code:

1. **Error Analysis**
   - Categorize errors (compilation, simulation, synthesis, etc.)
   - Identify error patterns and relationships between multiple errors
   - Determine if errors are syntax-related, interface-related, or algorithm-related

2. **Root Cause Identification**
   - Locate the specific code causing each error
   - Analyze context surrounding the problematic code
   - Identify patterns of misuse of HLS constructs or C++ language features
   - Check for common HLS pitfalls:
     - Unsupported C++ features in HLS
     - Memory access pattern issues
     - Data type incompatibilities
     - Interface specification problems
     - Pragma-related issues

3. **Solution Development**
   - Propose specific fixes for each identified issue
   - Provide explanations for why the fixes will resolve the errors
   - Include code snippets showing the corrections
   - Address any potential side effects of the proposed changes

4. **Verification Guidance**
   - Suggest verification steps to ensure the fixes are correct
   - Recommend additional tests if appropriate
   - Provide guidance on preventing similar issues in the future

## IMPORTANT: Response Format
1. First, provide your analysis of the issue
2. Then, clearly indicate the start of the corrected code with "### COMPLETE CORRECTED SOURCE CODE:"
3. Provide the ENTIRE corrected source code file in a single code block, not just the changes
4. If you have multiple files, provide each file in a separate code block
5. Use the following format for code blocks:
  - For function code file
    **File: `{component}.cpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For header file
    **File: `{component}.hpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For test bench file
    **File: `{component}_tb.cpp`**
    
    ```cpp
    // Your complete corrected code here
    ```
"""

def update_debug_prompt_template(template, error_info, result_status):
    """Update the debug prompt template with new information from this debugging session."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    prompt_dir = os.path.join(project_dir, "prompts")
    template_path = os.path.join(prompt_dir, "hls_debugging.md")
    
    # Create prompts directory if it doesn't exist
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Create backup before updating
    if os.path.exists(template_path):
        backup_dir = os.path.join(prompt_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"hls_debugging_{timestamp}.md")
        shutil.copy2(template_path, backup_path)
    
    # Extract error patterns to add to the template
    error_patterns = []
    common_errors = [
        "array index out of bounds",
        "incompatible types",
        "undefined reference",
        "undeclared identifier",
        "segmentation fault",
        "memory access violation",
        "variable not initialized",
        "precision loss",
        "overflow",
        "interface mismatch"
    ]
    
    for err in common_errors:
        if err.lower() in error_info.lower():
            error_patterns.append(err)
    
    # Update the template content
    updated_content = template
    
    # If success, add to BEST PRACTICES section
    if result_status == "success":
        if "## BEST PRACTICES" not in updated_content:
            updated_content += "\n\n## BEST PRACTICES\n"
        
        # Find position to insert new practice
        if "## BEST PRACTICES" in updated_content:
            sections = updated_content.split("## BEST PRACTICES")
            next_section_pos = sections[1].find("##")
            
            if next_section_pos > 0:
                insert_pos = next_section_pos
                practice_text = f"\n- Successfully resolved debugging issues on {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
                sections[1] = sections[1][:insert_pos] + practice_text + sections[1][insert_pos:]
            else:
                practice_text = f"\n- Successfully resolved debugging issues on {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
                sections[1] += practice_text
                
            updated_content = "## BEST PRACTICES".join(sections)
    
    # If error patterns found, add to COMMON PITFALLS section
    elif error_patterns:
        if "## COMMON PITFALLS" not in updated_content:
            updated_content += "\n\n## COMMON PITFALLS\n"
        
        # Find position to insert new pitfalls
        if "## COMMON PITFALLS" in updated_content:
            sections = updated_content.split("## COMMON PITFALLS")
            next_section_pos = sections[1].find("##")
            
            pitfalls_text = ""
            for pattern in error_patterns:
                pitfalls_text += f"\n- Watch for '{pattern}' errors which appeared in this debugging session\n"
            
            if next_section_pos > 0:
                insert_pos = next_section_pos
                sections[1] = sections[1][:insert_pos] + pitfalls_text + sections[1][insert_pos:]
            else:
                sections[1] += pitfalls_text
                
            updated_content = "## COMMON PITFALLS".join(sections)
    
    # Ensure the IMPORTANT section remains unchanged
    important_section = """## IMPORTANT: Response Format
1. First, provide your analysis of the issue
2. Then, clearly indicate the start of the corrected code with "### COMPLETE CORRECTED SOURCE CODE:"
3. Provide the ENTIRE corrected source code file in a single code block, not just the changes
4. If you have multiple files, provide each file in a separate code block
5. Use the following format for code blocks:
  - For function code file
    **File: `{component}.cpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For header file
    **File: `{component}.hpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For test bench file
    **File: `{component}_tb.cpp`**
    
    ```cpp
    // Your complete corrected code here
    ```"""
    
    # Check if the important section needs to be preserved/restored
    if "## IMPORTANT: Response Format" in updated_content:
        parts = updated_content.split("## IMPORTANT: Response Format")
        first_part = parts[0]
        # Find the next section after IMPORTANT
        next_part_pos = parts[1].find("\n## ")
        if next_part_pos > 0:
            remaining_part = parts[1][next_part_pos:]
            updated_content = first_part + important_section + remaining_part
        else:
            updated_content = first_part + important_section
    
    # Write the updated template
    with open(template_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated debug prompt template with new information at: {template_path}")
    return True

def create_debug_prompt(error_info, source_files_content, template=None):
    """Create a well-structured debug prompt for the LLM."""
    if template is None:
        template = get_debug_prompt_template()
    
    source_code_sections = []
    
    for file_path, content in source_files_content.items():
        source_code_sections.append(f"File: `{file_path}`:\n\n```cpp\n{content}\n```\n")
    
    all_source_code = "\n".join(source_code_sections)
    
    # Replace placeholders in the template
    prompt = template.replace("{{SOURCE_FILES}}", all_source_code)
    prompt = prompt.replace("{{ERROR_LOG}}", error_info)
    
    return prompt

def query_gemini(prompt, model="gemini-2.5-pro-exp-03-25"):
    """Send a prompt to Google Gemini API and get the response."""
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    # Extract the model name for the URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "key": GEMINI_API_KEY
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2
        }
    }
    
    try:
        response = requests.post(url, headers=headers, params=params, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        sys.exit(1)

def query_claude(prompt, model="claude-sonnet"):
    """Send a prompt to Anthropic Claude API and get the response."""
    if not CLAUDE_API_KEY:
        print("Error: CLAUDE_API_KEY environment variable not set.")
        sys.exit(1)
        
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Claude API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        sys.exit(1)

def query_openai(prompt, model="gpt-4"):
    """Send a prompt to OpenAI API and get the response."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2  # Lower temperature for more deterministic, focused responses
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        sys.exit(1)

def query_llm(prompt, model="gemini"):
    """Route the query to the appropriate LLM API based on the model."""
    if model.startswith("gemini"):
        return query_gemini(prompt, model)
    elif model.startswith("gpt"):
        return query_openai(prompt, model)
    elif model.startswith("claude"):
        return query_claude(prompt, model)
    else:
        print(f"Error: Unsupported model {model}.")
        sys.exit(1)

def save_to_markdown(source_files, error_info, response, model_name):
    """Save the debugging session to a markdown file."""
    # Create output filename based on first source file
    source_path = Path(source_files[0])
    output_dir = source_path.parent / "debug_reports"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{source_path.stem}_debug_report_{timestamp}.md"
    
    # Format the content
    files_list = "\n".join([f"- `{f}`" for f in source_files])
    
    content = f"""# Debug Report

## Error Information
```
{error_info}
```

## LLM Analysis and Suggestions ({model_name})
{response}

## Source Files
{files_list}

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    return output_file

def parse_code_corrections(response):
    """Parse the LLM response to extract code corrections with their file names."""
    file_code_blocks = {}
    
    # Match patterns like '**File: `filename.cpp`**' or '**File: `filename.hpp`**' followed by code blocks
    file_header_patterns = [
        # Match markdown-style headers: **File: `filename.cpp`**
        r'\*\*File:\s*`([^`]+)`\*\*.*?```(?:\w+)?\n(.*?)```',
        # Match header format: # filename.cpp or ## filename.cpp
        r'#{1,3}\s+([a-zA-Z0-9_]+\.[ch]pp).*?```(?:\w+)?\n(.*?)```',
        # Match "File:" or "Filename:" in headings followed by code blocks
        r'(?:File|Filename):\s*([a-zA-Z0-9_]+\.[ch]pp).*?```(?:\w+)?\n(.*?)```',
        # Match file headers with code blocks where header is after a new line
        r'\n([a-zA-Z0-9_]+\.(?:cpp|hpp|h))\n+```(?:\w+)?\n(.*?)```'
    ]
    
    # Apply each pattern to extract file headers and code blocks
    for pattern in file_header_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for filename, code in matches:
            clean_filename = filename.strip()
            if clean_filename and clean_filename not in file_code_blocks:
                file_code_blocks[clean_filename] = code.strip()
    
    # Extract code blocks with filename comments at beginning
    comment_patterns = [
        # Match C++-style comments: // File: filename.cpp or // filename.cpp:
        r'```(?:\w+)?\n\s*(?://|/\*)\s*(?:File|Filename)?:?\s*([a-zA-Z0-9_]+\.[ch]pp).*?\n(.*?)```',
        # Match filename followed by code where filename is in comments
        r'```(?:\w+)?\n\s*(?://|/\*)\s*([a-zA-Z0-9_]+\.[ch]pp).*?\n(.*?)```'
    ]
    
    for pattern in comment_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for filename, code in matches:
            clean_filename = filename.strip()
            if clean_filename and clean_filename not in file_code_blocks:
                # Remove the first line with the comment if it's there
                code_lines = code.split('\n')
                if '// File:' in code_lines[0] or '/* File:' in code_lines[0]:
                    code = '\n'.join(code_lines[1:])
                file_code_blocks[clean_filename] = code.strip()
    
    # If regex didn't find anything, try line-by-line approach
    if not file_code_blocks:
        # Look for "COMPLETE CORRECTED SOURCE CODE:" section
        complete_section_match = re.search(r'###\s*COMPLETE\s*CORRECTED\s*SOURCE\s*CODE:.*?$(.*)', response, re.DOTALL | re.IGNORECASE)
        if complete_section_match:
            complete_section = complete_section_match.group(1)
            
            # Line-by-line extraction
            lines = complete_section.split('\n')
            current_file = None
            collecting = False
            current_block = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Look for file headers in markdown format
                if "**File:" in line and "`" in line:
                    start_idx = line.find('`') + 1
                    end_idx = line.rfind('`')
                    if start_idx > 0 and end_idx > start_idx:
                        current_file = line[start_idx:end_idx].strip()
                        i += 1
                        # Skip until code block start
                        while i < len(lines) and not (lines[i].startswith('```')):
                            i += 1
                        if i < len(lines):
                            i += 1  # Skip the ```cpp line
                            current_block = []
                            # Collect until code block end
                            while i < len(lines) and not lines[i].startswith('```'):
                                current_block.append(lines[i])
                                i += 1
                            if current_file and current_block:
                                file_code_blocks[current_file] = '\n'.join(current_block)
                            i += 1  # Skip the ``` line
                            continue
                
                # Check for file headers in headings
                if line.strip().startswith('#') and ('.cpp' in line or '.hpp' in line):
                    parts = line.strip('#').strip().split()
                    for part in parts:
                        if part.endswith('.cpp') or part.endswith('.hpp') or part.endswith('.h'):
                            current_file = part
                            break
                
                # Traditional code block extraction
                if line.startswith('```') and not collecting:
                    collecting = True
                    current_block = []
                    i += 1
                    continue
                elif line.startswith('```') and collecting:
                    collecting = False
                    if current_file and current_block:
                        file_code_blocks[current_file] = '\n'.join(current_block)
                        current_block = []
                    i += 1
                    continue
                    
                if collecting:
                    # Check for file indicators in comments
                    if line.strip().startswith('// File:') or line.strip().startswith('// filename:'):
                        current_file = line.split(':', 1)[1].strip()
                        # Skip this line from the code block
                        i += 1
                        continue
                    current_block.append(line)
                
                i += 1
    
    # If still no file blocks found, look for raw code blocks
    if not file_code_blocks:
        # Find any code blocks
        raw_code_blocks = re.findall(r'```(?:cpp|c\+\+)?\n(.*?)```', response, re.DOTALL)
        if raw_code_blocks:
            # If just one code block, use generic key
            file_code_blocks["corrected_code"] = raw_code_blocks[0].strip()
    
    return file_code_blocks

def edit_source_file(source_files, code_blocks):
    """Apply the suggested corrections directly to the source files after creating backups."""
    if not code_blocks:
        print("No code corrections found in the LLM response.")
        return False
    
    backup_files = []
    updated_files = []
    
    # First, make backups of all source files
    for source_file in source_files:
        source_path = Path(source_file)
        backup_file = source_path.with_suffix(f"{source_path.suffix}.bak")
        shutil.copy2(source_file, backup_file)
        backup_files.append(backup_file)
        print(f"Backup created: {backup_file}")
    
    # Process each source file and find matching code blocks
    for source_file in source_files:
        source_path = Path(source_file)
        source_filename = source_path.name
        source_stem = source_path.stem
        source_ext = source_path.suffix
        
        # Try exact filename match first
        if source_filename in code_blocks:
            with open(source_file, 'w') as f:
                f.write("/* AUTO-EDITED BY DEBUG ASSISTANT */\n")
                f.write(code_blocks[source_filename])
            updated_files.append(source_file)
            print(f"\nSource file {source_file} updated with suggested changes (exact filename match).")
            continue
        
        # Try to match by basename only (e.g., "peakPicker.cpp" vs "filepath/peakPicker.cpp")
        basename_matches = [k for k in code_blocks.keys() if Path(k).name == source_filename]
        if len(basename_matches) == 1:
            with open(source_file, 'w') as f:
                f.write("/* AUTO-EDITED BY DEBUG ASSISTANT */\n")
                f.write(code_blocks[basename_matches[0]])
            updated_files.append(source_file)
            print(f"\nSource file {source_file} updated with suggested changes (basename match).")
            continue
            
        # Try to match by file stem and extension
        for code_filename in code_blocks.keys():
            code_path = Path(code_filename)
            if code_path.stem == source_stem and code_path.suffix == source_ext:
                with open(source_file, 'w') as f:
                    f.write("/* AUTO-EDITED BY DEBUG ASSISTANT */\n")
                    f.write(code_blocks[code_filename])
                updated_files.append(source_file)
                print(f"\nSource file {source_file} updated with suggested changes (stem/ext match).")
                break
        else:  # This else belongs to the for loop, executes if no break occurred
            # If we get here, we didn't find a match by stem/ext
            
            # Try to match by extension only (if there's exactly one match)
            matching_ext_keys = [k for k in code_blocks.keys() if Path(k).suffix == source_ext]
            if len(matching_ext_keys) == 1:
                with open(source_file, 'w') as f:
                    f.write("/* AUTO-EDITED BY DEBUG ASSISTANT */\n")
                    f.write(code_blocks[matching_ext_keys[0]])
                updated_files.append(source_file)
                print(f"\nSource file {source_file} updated with suggested changes (extension match).")
                continue
            
            # Last resort: If we have a single source file and a generic code block
            if len(source_files) == 1 and "corrected_code" in code_blocks:
                with open(source_file, 'w') as f:
                    f.write("/* AUTO-EDITED BY DEBUG ASSISTANT */\n")
                    f.write(code_blocks["corrected_code"])
                updated_files.append(source_file)
                print(f"\nSource file {source_file} updated with generic correction.")
                continue
            
            print(f"Warning: No matching code block found for {source_file}")
    
    # If we still haven't found matches for all source files, try more permissive matching
    if len(updated_files) < len(source_files):
        for source_file in source_files:
            if source_file in updated_files:
                continue  # Skip already updated files
                
            source_path = Path(source_file)
            source_stem = source_path.stem
            
            # Try partial match - if a code block key contains the source stem
            stem_matches = [k for k in code_blocks.keys() if source_stem in k]
            if len(stem_matches) == 1:
                with open(source_file, 'w') as f:
                    f.write("/* AUTO-EDITED BY DEBUG ASSISTANT */\n")
                    f.write(code_blocks[stem_matches[0]])
                updated_files.append(source_file)
                print(f"\nSource file {source_file} updated with suggested changes (partial stem match).")
                continue
    
    if not updated_files:
        print("Warning: No files were updated. Could not match corrections to source files.")
    
    return backup_files

def main():
    """Main function to run the debug assistant."""
    args = parse_arguments()
    
    # Read the error log
    error_log = read_file(args.error_log)
    
    # Read multiple source files
    source_files_content = {}
    for source_file in args.source_file:
        source_files_content[source_file] = read_file(source_file)
    
    # Extract relevant error information
    error_info = extract_error_information(error_log)
    
    # Get prompt template (from file or default)
    prompt_template = None
    if args.prompt_template and os.path.exists(args.prompt_template):
        prompt_template = read_file(args.prompt_template)
    else:
        prompt_template = get_debug_prompt_template()
    
    # Create the debug prompt with multiple source files
    prompt = create_debug_prompt(error_info, source_files_content, prompt_template)
    
    print(f"Analyzing error using {args.model} and generating debug suggestions...")
    print("This may take a moment...")
    
    # Query the LLM for debugging help
    response = query_llm(prompt, args.model)
    
    # Save the response to a markdown file
    md_file = save_to_markdown(args.source_file, error_info, response, args.model)
    print(f"\nDebug report saved to: {md_file}")
    
    # Parse code corrections from the response
    code_blocks = parse_code_corrections(response)
    
    # Print the formatted response
    print("\n" + "="*80)
    print("DEBUG ASSISTANT SUGGESTIONS")
    print("="*80 + "\n")
    # print(response)
    print("\n" + "="*80)
    
    # Automatically apply the suggested fixes if code blocks are found
    result_status = "failed"
    if code_blocks:
        backup_files = edit_source_file(args.source_file, code_blocks)
        print(f"Changes applied automatically. Original sources backed up.")
        result_status = "success"
    else:
        print("No specific code corrections found in the LLM response.")
    
    # Update the debug prompt template with the results of this session
    update_debug_prompt_template(prompt_template, error_info, result_status)
    
    print("\nC simulation can now be re-run with the updated source files.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
````

Now that I've updated the `debug_assistant.py` file to use `hls_debugging.md` instead of `enhanced_debug_assistant.md`, the system will automatically:

1. Look for the prompt template in `hls_debugging.md` first
2. Create this file with a default template if it doesn't exist
3. Make backups of the prompt file before updating it
4. Update the prompt with new information from debugging sessions
5. Preserve the important response format section as specified

The changes ensure that the system continues to learn and improve the prompt through debugging sessions while maintaining the critical format section needed for proper parsing of the LLM's responses.

