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

def create_debug_prompt(error_info, source_files_content):
    """Create a well-structured debug prompt for the LLM."""
    source_code_sections = []
    
    for file_path, content in source_files_content.items():
        source_code_sections.append(f"File: `{file_path}`:\n\n```cpp\n{content}\n```\n")
    
    all_source_code = "\n".join(source_code_sections)
    
    prompt = f"""
# HLS C++ Debug Request

## Error Information
I'm encountering errors in my High-Level Synthesis (HLS) C++ implementation. 
The C simulation has failed with the following errors:

```
{error_info}
```

## Source Code
{all_source_code}

## Assistance Needed
Please help me debug these HLS C++ code files and its associated `*.hpp` header file by:

1. Analyzing the error message to understand the root cause
2. Examining the source code for issues related to the error
3. Providing a complete corrected version of the source file
After your analysis, provide the COMPLETE corrected source code file that resolves the issue.

Focus on common HLS issues like:
- Mismatches between test bench and implementation interfaces
- Data type incompatibilities
- Array indexing errors
- Logical errors in the algorithm implementation
- Pragma-related issues

IMPORTANT: Response Format
1. First, provide your analysis of the issue
2. Then, clearly indicate the start of the corrected code with "### COMPLETE CORRECTED SOURCE CODE:"
3. Provide the ENTIRE corrected source code file in a single code block, not just the changes
4. If you have multiple files, provide each file in a separate code block
5. Use the following format for code blocks:
  - For function code file
    **File: `peakPicker.cpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For header file
    **File: `peakPicker.hpp`**

    ```cpp
    // Your complete corrected code here
    ```
  - For test bench file
    **File: `peakPicker_tb.cpp`**
    
    ```cpp
    // Your complete corrected code here
    ```
"""
    return prompt

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
    
    # Create the debug prompt with multiple source files
    prompt = create_debug_prompt(error_info, source_files_content)
    
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
    if code_blocks:
        backup_files = edit_source_file(args.source_file, code_blocks)
        print(f"Changes applied automatically. Original sources backed up.")
    else:
        print("No specific code corrections found in the LLM response.")
    
    print("\nC simulation can now be re-run with the updated source files.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
