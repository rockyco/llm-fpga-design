#!/usr/bin/env python3

import argparse
import os
import sys
import requests
import json
import re
import openai
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables for API keys
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate HLS C++ code from MATLAB reference using LLM')
    parser.add_argument('--matlab_file', nargs='+', required=True, 
                        help='Path to MATLAB reference file(s)')
    parser.add_argument('--prompt', required=True, 
                        help='Path to prompt template file')
    parser.add_argument('--output_dir', default='implementations', 
                        help='Directory to save generated HLS code')
    parser.add_argument('--model', default='gemini-2.0-pro-exp', 
                        help='LLM model to use (default: gemini-2.0-pro-exp)')
    parser.add_argument('--api_key', 
                        help='API key for LLM service (or set GEMINI_API_KEY environment variable)')
    return parser.parse_args()

def read_file(file_path):
    """Read and return the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def create_prompt(matlab_files, prompt_template):
    """Create the final prompt by combining MATLAB code with the template."""
    # Get component name from first MATLAB file
    component_name = os.path.basename(matlab_files[0]).split('.')[0]
    if component_name.endswith('_tb'):
        component_name = component_name[:-3]  # Remove _tb suffix
        
    # Create MATLAB code section
    matlab_code = ""
    for file_path in matlab_files:
        matlab_code += f"\n## File: {os.path.basename(file_path)}\n```matlab\n{read_file(file_path)}\n```\n"
    
    # Add specific instructions for testbench generation
    testbench_instructions = """
Please generate the following files:
1. A header file (*.hpp) with the appropriate declarations
2. An implementation file (*.cpp) with the HLS implementation
3. A testbench file (*_tb.cpp) that:
   - Reads input data from *_in.txt files
   - Compares output with reference data from *_ref.txt files
   - Outputs results to *_out.txt files
   - Includes proper verification and error reporting

Follow the structure of the example files provided."""
    
    # Replace template variables including component name
    if "{{component}}" in prompt_template:
        prompt_template = prompt_template.replace("{{component}}", component_name)
    elif "{component}" in prompt_template:
        prompt_template = prompt_template.replace("{component}", component_name)
    
    # Check for MATLAB code placeholder
    if "MATLAB_CODE" in prompt_template or "{{MATLAB_CODE}}" in prompt_template:
        # Replace placeholder in template if it exists
        if "{{MATLAB_CODE}}" in prompt_template:
            prompt = prompt_template.replace("{{MATLAB_CODE}}", matlab_code)
        else:
            # Handle older format without braces
            prompt = prompt_template.replace("MATLAB_CODE", matlab_code)
    else:
        # Append MATLAB code for backward compatibility
        prompt = f"{prompt_template}\n\n# MATLAB Reference Implementation\n{matlab_code}"
    
    # Add testbench instructions if not already in the template
    if "testbench" not in prompt.lower() or "tb.cpp" not in prompt.lower():
        prompt += "\n\n" + testbench_instructions
    
    return prompt

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
        "messages": [
            {"role": "system", "content": "You are an expert FPGA developer specializing in HLS C++ implementations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1  # Lower temperature for more deterministic output
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
        "temperature": 0.1
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

def query_gemini(prompt, model="gemini-2.0-pro-exp"):
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
        "contents": [{"parts": [{"text": "You are an expert FPGA developer specializing in HLS C++ implementations.\n\n" + prompt}]}],
        "generationConfig": {
            "temperature": 0.1
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

def query_llm(prompt, model="gemini-2.0-pro-exp"):
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

def extract_code_blocks(llm_response, component_name):
    """Extract code blocks from LLM response."""
    code_blocks = {}
    
    # Use regex to extract code blocks with explicit file headers
    file_header_patterns = [
        # Match markdown-style headers: **File: `filename.cpp`**
        r'\*\*File:\s*`([^`]+)`\*\*.*?```(?:\w+)?\n(.*?)```',
        # Match header format: # filename.cpp or ## filename.cpp
        r'#{1,3}\s+([a-zA-Z0-9_]+\.[ch]pp).*?```(?:\w+)?\n(.*?)```',
        # Match "File:" or "Filename:" in headings followed by code blocks
        r'(?:File|Filename):\s*([a-zA-Z0-9_]+\.[ch]pp).*?```(?:\w+)?\n(.*?)```',
        # Match file headers with code blocks where header is after a new line
        r'\n([a-zA-Z0-9_]+\.(?:cpp|hpp|h))\n+```(?:\w+)?\n(.*?)```',
        # Match testbench headers specifically
        r'(?:Test[Bb]ench|TB) (?:File|Code)?:?\s*.*?```(?:\w+)?\n(.*?)```'
    ]
    
    # Apply each regex pattern to extract file headers and code blocks
    for pattern in file_header_patterns:
        if "Test" in pattern or "TB" in pattern:
            # Special case for testbench without filename
            matches = re.findall(pattern, llm_response, re.DOTALL)
            if matches:
                code_blocks[f"{component_name}_tb.cpp"] = matches[0]
        else:
            matches = re.findall(pattern, llm_response, re.DOTALL)
            for filename, code in matches:
                clean_filename = filename.strip()
                if clean_filename and clean_filename not in code_blocks:
                    code_blocks[clean_filename] = code
    
    # Extract code blocks with filename comments at beginning
    comment_patterns = [
        # Match C++-style comments: // File: filename.cpp or // filename.cpp:
        r'```(?:\w+)?\n\s*(?://|/\*)\s*(?:File|Filename)?:?\s*([a-zA-Z0-9_]+\.[ch]pp).*?\n(.*?)```',
        # Match filename followed by code where filename is in comments
        r'```(?:\w+)?\n\s*(?://|/\*)\s*([a-zA-Z0-9_]+\.[ch]pp).*?\n(.*?)```'
    ]
    
    for pattern in comment_patterns:
        matches = re.findall(pattern, llm_response, re.DOTALL)
        for filename, code in matches:
            clean_filename = filename.strip()
            if clean_filename and clean_filename not in code_blocks:
                # Remove the first line with the comment if it's there
                code_lines = code.split('\n')
                if '// File:' in code_lines[0] or '/* File:' in code_lines[0]:
                    code = '\n'.join(code_lines[1:])
                code_blocks[clean_filename] = code
    
    # If regex didn't find anything, revert to line-by-line approach
    if not code_blocks:
        lines = llm_response.split('\n')
        
        # Line-by-line extraction
        current_file = None
        collecting = False
        current_block = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for file headers in markdown format: **File: `filename`**
            if "**File:" in line and "`" in line:
                # Extract filename from pattern **File: `filename`**
                start_idx = line.find('`') + 1
                end_idx = line.rfind('`')
                if start_idx > 0 and end_idx > start_idx:
                    current_file = line[start_idx:end_idx].strip()
                    i += 1
                    # Skip until we find the code block start
                    while i < len(lines) and not (lines[i].startswith('```')):
                        i += 1
                    if i < len(lines):
                        i += 1  # Skip the ```cpp or ```c++ line
                        current_block = []
                        # Collect until code block end
                        while i < len(lines) and not lines[i].startswith('```'):
                            current_block.append(lines[i])
                            i += 1
                        if current_file and current_block:
                            code_blocks[current_file] = '\n'.join(current_block)
                        i += 1  # Skip the ``` line
                        continue
            
            # Check for file headers in headings
            if line.strip().startswith('#') and '.cpp' in line or '.hpp' in line:
                # Try to extract filename
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
                    code_blocks[current_file] = '\n'.join(current_block)
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
    
    # Expected files based on component name
    expected_files = [
        f"{component_name}.hpp",
        f"{component_name}.cpp",
        f"{component_name}_tb.cpp"
    ]
    
    # If we didn't find all expected files, try content-based detection as backup
    if not all(file in code_blocks for file in expected_files):
        # Find code blocks without specific filenames that might match our expected files
        unnamed_blocks = []
        
        # First collect all code blocks that don't have a filename
        raw_code_blocks = re.findall(r'```(?:cpp|c\+\+)?\n(.*?)```', llm_response, re.DOTALL)
        for block in raw_code_blocks:
            if not any(block in content for content in code_blocks.values()):
                unnamed_blocks.append(block)
        
        # Try to classify these unnamed blocks based on content patterns
        for block in unnamed_blocks:
            # Check for header file patterns
            is_header = '#ifndef' in block or '#pragma once' in block or \
                        (('class' in block or 'struct' in block) and \
                         ('};' in block) and \
                         not ('int main' in block))
            
            # Check for implementation file patterns
            is_implementation = (f'#include "{component_name}.hpp"' in block or \
                               f'void {component_name}' in block or \
                               'hls::' in block) and \
                              not ('int main' in block)
            
            # Enhanced testbench detection
            is_testbench = ('int main' in block and \
                          ('test' in block.lower() or 'compare' in block.lower() or 
                           'verify' in block.lower() or 'read' in block.lower() and 'write' in block.lower())) \
                          or f'#include "{component_name}.hpp"' in block and 'int main' in block
            
            # Assign to appropriate file
            if is_header and not any(file.endswith('.hpp') for file in code_blocks):
                code_blocks[f'{component_name}.hpp'] = block
            elif is_implementation and not any(file.endswith('.cpp') and not file.endswith('_tb.cpp') for file in code_blocks):
                code_blocks[f'{component_name}.cpp'] = block
            elif is_testbench and not any(file.endswith('_tb.cpp') for file in code_blocks):
                code_blocks[f'{component_name}_tb.cpp'] = block
    
    # Additional testbench extraction as final fallback
    if f'{component_name}_tb.cpp' not in code_blocks:
        for block in raw_code_blocks:
            if 'int main' in block and (f'{component_name}' in block) and block not in code_blocks.values():
                code_blocks[f'{component_name}_tb.cpp'] = block
                break
    
    # If we found the implementation but not the header, look for header-like content in the implementation
    if f'{component_name}.cpp' in code_blocks and f'{component_name}.hpp' not in code_blocks:
        cpp_content = code_blocks[f'{component_name}.cpp']
        
        # Find header-like sections in the cpp file
        if '#ifndef' in cpp_content or '#pragma once' in cpp_content:
            # Split at the first implementation indicator
            split_markers = ["#include <", "void ", "int ", "float ", "double "]
            split_positions = []
            
            for marker in split_markers:
                if marker in cpp_content:
                    # Find position but skip header guards/includes by checking for non-commented occurrence
                    lines = cpp_content.split('\n')
                    for i, line in enumerate(lines):
                        if marker in line and not line.strip().startswith('//'):
                            if marker == '#include <' and i > 10:  # Only consider includes after the first few lines as implementation
                                split_positions.append(cpp_content.find(line))
                                break
                            elif marker != '#include <':
                                split_positions.append(cpp_content.find(line))
                                break
            
            if split_positions:
                split_pos = min(split_positions)
                header_content = cpp_content[:split_pos].strip()
                impl_content = cpp_content[split_pos:].strip()
                
                # Only split if we have substantial content in both parts
                if len(header_content) > 50 and len(impl_content) > 50:
                    code_blocks[f'{component_name}.hpp'] = header_content
                    code_blocks[f'{component_name}.cpp'] = impl_content
    
    return code_blocks

def save_code_to_files(code_blocks, output_dir):
    """Save extracted code blocks to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    files_saved = []
    for filename, content in code_blocks.items():
        # Clean up filename if needed
        clean_filename = os.path.basename(filename.strip())
        file_path = os.path.join(output_dir, clean_filename)
        
        # Ensure content ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        files_saved.append(file_path)
        print(f"Saved: {file_path}")
    
    if not files_saved:
        print("Warning: No files were saved. Check the code block extraction logic.")
    
    return files_saved

def main():
    args = parse_arguments()
    
    # Load environment variables for API keys
    load_dotenv()
    
    # Set API key from args or environment variable
    api_key = args.api_key
    if api_key:
        if "gemini" in args.model.lower():
            os.environ['GEMINI_API_KEY'] = api_key
        elif "claude" in args.model.lower():
            os.environ['CLAUDE_API_KEY'] = api_key
        else:
            os.environ['OPENAI_API_KEY'] = api_key
    
    # Use only hls_generation.md prompt
    prompt_name = "hls_generation"
    
    # Read MATLAB files
    matlab_files = [file for file in args.matlab_file]
    
    # Get project directory path early to avoid undefined variable error
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Read prompt template - first check if we need to load from file
    if os.path.exists(args.prompt):
        prompt_template = read_file(args.prompt)
    else:
        # Try to find the prompt in standard locations
        prompts_dir = os.path.join(project_dir, "prompts")
        
        # Check for hls_generation.md - our standard prompt
        standard_prompt = os.path.join(prompts_dir, "hls_generation.md")
        if os.path.exists(standard_prompt):
            prompt_template = read_file(standard_prompt)
            prompt_name = "hls_generation"
        else:
            # Fallback to the provided prompt file
            prompt_template = read_file(args.prompt)
    
    # Create the full prompt
    full_prompt = create_prompt(matlab_files, prompt_template)
    
    # Determine output directory and component name - remove any _tb suffix
    component_name = os.path.basename(matlab_files[0]).split('.')[0]
    if component_name.endswith('_tb'):
        component_name = component_name[:-3]  # Remove _tb suffix
    
    output_dir = os.path.join(args.output_dir, component_name)
    
    print(f"Generating HLS code for {component_name}...")
    print(f"Using model: {args.model}")
    print(f"Using prompt template: {prompt_name}")
    
    # Use the unified query_llm function to call the appropriate API
    llm_response = query_llm(full_prompt, args.model)
    
    # Extract code blocks from response
    code_blocks = extract_code_blocks(llm_response, component_name)
    
    if not code_blocks:
        print("Warning: No code blocks detected in the LLM response.")
        # Save the full response as a reference
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "llm_response.md"), 'w') as f:
            f.write(llm_response)
        
        # Track error for prompt improvement
        with open(os.path.join(project_dir, "prompt_feedback.json"), "a") as f:
            feedback = {
                "prompt": prompt_name,
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "component": component_name,
                "status": "failed",
                "error": "No code blocks detected in response"
            }
            f.write(json.dumps(feedback) + "\n")
            
        sys.exit(1)
    
    # Save code to files
    saved_files = save_code_to_files(code_blocks, output_dir)
    
    print("\nHLS code generation complete!")
    print(f"Files generated: {len(saved_files)}")
    print(f"Output directory: {output_dir}")
    
    # Save the full LLM response for reference
    with open(os.path.join(output_dir, "llm_response.md"), 'w') as f:
        f.write(llm_response)
    
    # Track success for prompt improvement
    with open(os.path.join(project_dir, "prompt_feedback.json"), "a") as f:
        feedback = {
            "prompt": prompt_name,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "component": component_name,
            "status": "success",
            "files_generated": len(saved_files),
            "file_types": list(code_blocks.keys())
        }
        f.write(json.dumps(feedback) + "\n")
    
    print("\nNext steps:")
    print(f"cd {output_dir}")
    print("make csim  # Run C simulation")
    
    # Return the output directory for the orchestrator
    print(output_dir)

if __name__ == "__main__":
    main()
