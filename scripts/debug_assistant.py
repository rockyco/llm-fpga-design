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
    parser.add_argument('--model', type=str, default='gemini-2.0-pro-exp',
                        choices=['gemini-2.0-pro-exp', 'gemini-2.0-flash-thinking-exp', 'gpt-4', 'gpt-3.5-turbo', 'claude-sonnet'],
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
Please help me debug this HLS C++ code by:

1. Analyzing the error message to understand the root cause
2. Examining the source code for issues related to the error
3. Suggesting specific code changes to fix the problem
4. Explaining why these changes will address the error

Focus on common HLS issues like:
- Mismatches between test bench and implementation interfaces
- Data type incompatibilities
- Array indexing errors
- Logical errors in the algorithm implementation
- Pragma-related issues

Please format your response with a clear analysis followed by specific code fixes.
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
    """Parse the LLM response to extract code corrections, prioritizing the final/refined version."""
    
    # First, try to find code blocks specifically labeled as refined or final corrections
    refined_markers = [
        r'Refined Correction.*?```(?:cpp)?\n(.*?)```',
        r'Final Solution.*?```(?:cpp)?\n(.*?)```',
        r'Recommended Fix.*?```(?:cpp)?\n(.*?)```',
        r'Better Solution.*?```(?:cpp)?\n(.*?)```'
    ]
    
    for marker in refined_markers:
        refined_blocks = re.findall(marker, response, re.DOTALL | re.IGNORECASE)
        if refined_blocks:
            return [refined_blocks[0]]  # Return the refined solution as highest priority
    
    # If no refined blocks found, look for any code blocks labeled as "corrected" or "fixed"
    corrected_markers = [
        r'Corrected Code.*?```(?:cpp)?\n(.*?)```',
        r'Fixed Code.*?```(?:cpp)?\n(.*?)```'
    ]
    
    for marker in corrected_markers:
        corrected_blocks = re.findall(marker, response, re.DOTALL | re.IGNORECASE)
        if corrected_blocks:
            return [corrected_blocks[0]]  # Return the corrected solution
    
    # Fall back to the original behavior
    code_blocks = re.findall(r'```cpp\n(.*?)```', response, re.DOTALL)
    
    # If no code blocks with explicit cpp tag, try to find any code blocks
    if not code_blocks:
        code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    
    return code_blocks

def edit_source_file(source_files, code_blocks):
    """Save the suggested corrections to separate files without modifying the originals."""
    if not code_blocks:
        print("No code corrections found in the LLM response.")
        return False
    
    backup_files = []
    # Make backups of all original files
    for source_file in source_files:
        source_path = Path(source_file)
        backup_file = source_path.with_suffix(f"{source_path.suffix}.bak")
        shutil.copy2(source_file, backup_file)
        backup_files.append(backup_file)
    
    print("\nSuggested code correction:")
    print("-" * 40)
    for i, block in enumerate(code_blocks):
        print(f"Code Block {i+1}:")
        print(block)
        print("-" * 40)
    
    # Save suggestions to a separate file (based on first source file)
    source_path = Path(source_files[0])
    suggestion_file = source_path.with_name(f"{source_path.stem}_suggested{source_path.suffix}")
    with open(suggestion_file, 'w') as f:
        f.write("/* SUGGESTED CHANGES FROM DEBUG ASSISTANT */\n\n")
        for i, block in enumerate(code_blocks):
            f.write(f"/* SUGGESTION {i+1} */\n")
            f.write(block)
            f.write("\n\n")
    
    print(f"Suggestions saved to: {suggestion_file}")
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
    print(response)
    print("\n" + "="*80)
    
    # Ask user if they want to apply the suggested fixes
    if code_blocks:
        apply_changes = input("Do you want to apply the suggested code changes? (y/n): ").lower().strip()
        if apply_changes == 'y':
            backup_files = edit_source_file(args.source_file, code_blocks)
            print(f"Changes applied. Original sources backed up.")
            print("The edited sections are marked with /* AUTO-EDITED BY DEBUG ASSISTANT */ comments.")
        else:
            print("No changes applied. You can manually edit the source files based on the suggestions.")
    else:
        print("No specific code corrections found in the LLM response.")
    
    print("\nTo apply these fixes manually, edit your source files and re-run C simulation.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
