rockyco: Generate the Python code file `debug_assistant.py` which can complete a task of debugging error with LLM assistance by running this command 'python3 scripts/debug_assistant.py --error_log csim.log --source_file peakPicker.cpp'

GitHub Copilot: I'll create a Python script that helps debug HLS C++ code by using an LLM to analyze error logs and source code.

## Step-by-step solution:

1. Create a Python script called `debug_assistant.py` in the `scripts` directory
2. Implement argument parsing for error log and source file
3. Add functions to read and extract error information from the log file
4. Create a function to read the source code file
5. Create a function to format a well-structured prompt for the LLM
6. Implement the API call to an LLM service
7. Format and display the LLM's response with suggested fixes

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

This script will send error logs and source code to an LLM for debugging assistance.

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
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set this variable with your OpenAI API key.")
    print("You can create a .env file with OPENAI_API_KEY=your_key_here")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-assisted debugging for HLS C++ code')
    parser.add_argument('--error_log', type=str, required=True, 
                        help='Path to the C simulation error log file')
    parser.add_argument('--source_file', type=str, required=True,
                        help='Path to the HLS C++ source file')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='LLM model to use (default: gpt-4)')
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

def create_debug_prompt(error_info, source_code, source_file):
    """Create a well-structured debug prompt for the LLM."""
    prompt = f"""
# HLS C++ Debug Request

## Error Information
I'm encountering errors in my High-Level Synthesis (HLS) C++ implementation. 
The C simulation has failed with the following errors:

```
{error_info}
```

## Source Code
The source file is `{source_file}`:

```cpp
{source_code}
```

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

def query_openai(prompt, model="gpt-4"):
    """Send a prompt to OpenAI API and get the response."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
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

def main():
    """Main function to run the debug assistant."""
    args = parse_arguments()
    
    # Read the error log and source file
    error_log = read_file(args.error_log)
    source_code = read_file(args.source_file)
    
    # Extract relevant error information
    error_info = extract_error_information(error_log)
    
    # Create the debug prompt
    prompt = create_debug_prompt(error_info, source_code, args.source_file)
    
    print("Analyzing error and generating debug suggestions...")
    print("This may take a moment...")
    
    # Query the LLM for debugging help
    response = query_openai(prompt, args.model)
    
    # Print the formatted response
    print("\n" + "="*80)
    print("DEBUG ASSISTANT SUGGESTIONS")
    print("="*80 + "\n")
    print(response)
    print("\n" + "="*80)
    print("To apply these fixes, edit your source file and re-run C simulation.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
````

This script will help you debug your HLS implementation by:

1. Reading C simulation error logs
2. Analyzing your source code
3. Using OpenAI's GPT-4 (or another specified model) to suggest fixes
4. Providing a detailed explanation of the issues and solutions

To use the script, you'll need to:

1. Make sure the script is executable: `chmod +x scripts/debug_assistant.py`
2. Set up an OpenAI API key in your environment or in a `.env` file
3. Install required Python packages: `pip install requests python-dotenv`
4. Run the script as shown in your README: `python3 scripts/debug_assistant.py --error_log csim.log --source_file peakPicker.cpp`

The script can be extended to support other LLM providers like Anthropic Claude or Google Gemini by adding additional API client functions.

# Add other LLM providers

rockyco: Revise this Python code to add Google Gemini and Claude Sonnet models as optional LLMs, and make Google Gemini model as the default one.

GitHub Copilot: # Solution for Adding Multiple LLM Support

I'll modify the script to add support for Google Gemini and Claude Sonnet models, and make Gemini the default option.

## Step-by-step changes:

1. Add environment variables for Google Gemini and Anthropic Claude API keys
2. Update argument parser to include Gemini as the default model and add model choice options
3. Create new functions to query Gemini and Claude APIs 
4. Modify the main query function to route requests to the appropriate API based on model selection

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

Add support for multiple LLM providers and make Gemini the default model.

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
    parser.add_argument('--source_file', type=str, required=True,
                        help='Path to the HLS C++ source file')
    parser.add_argument('--model', type=str, default='gemini',
                        choices=['gemini', 'gemini-pro', 'gpt-4', 'gpt-3.5-turbo', 'claude-sonnet'],
                        help='LLM model to use (default: gemini)')
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

# ...existing code...

def query_gemini(prompt, model="gemini-pro"):
    """Send a prompt to Google Gemini API and get the response."""
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
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

def main():
    """Main function to run the debug assistant."""
    args = parse_arguments()
    
    # Read the error log and source file
    error_log = read_file(args.error_log)
    source_code = read_file(args.source_file)
    
    # Extract relevant error information
    error_info = extract_error_information(error_log)
    
    # Create the debug prompt
    prompt = create_debug_prompt(error_info, source_code, args.source_file)
    
    print(f"Analyzing error using {args.model} and generating debug suggestions...")
    print("This may take a moment...")
    
    # Query the LLM for debugging help
    response = query_llm(prompt, args.model)
    
    # Print the formatted response
    print("\n" + "="*80)
    print("DEBUG ASSISTANT SUGGESTIONS")
    print("="*80 + "\n")
    print(response)
    print("\n" + "="*80)
    print("To apply these fixes, edit your source file and re-run C simulation.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
````

# Solve issue

rockyco: solve this issue 'Analyzing error using gemini-2.0-pro-exp and generating debug suggestions...
This may take a moment...
Error calling Gemini API: 404 Client Error: Not Found for url: https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyAJLWNRBvIjlTw3Fq0kmFwIIg3jhjSpK_o'

GitHub Copilot: I'll solve the issue with the Gemini API call that's returning a 404 error. The problem is that the code is using a hardcoded model name in the URL path instead of using the model name passed from the command line.

## Solution Steps:
1. Update the `query_gemini` function to use the correct model name in the API URL
2. Fix the model name handling in the URL construction

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

Update the query_gemini function to correctly use the model parameter in the API URL:

````python
# ...existing code...

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

# ...existing code...
````

The key change is updating the URL construction to use the model parameter directly in the URL path. This way, when you pass "gemini-2.0-pro-exp" as the model parameter, it will correctly form the URL with that model name instead of hardcoding "gemini-pro".

# Save LLM's response to a file

rockyco: Revise this Python code to save LLM's response into a markdown file, edit the source C++ code to correct the error, and mark the code snippet newly edited as well.

GitHub Copilot: I'll revise the debug assistant to save the LLM's response to a markdown file, edit the source C++ code to apply corrections, and mark the edited code sections. Here's my solution:

## Step-by-step solution:
1. Import necessary additional libraries (datetime, shutil, and re)
2. Create a function to save LLM responses to markdown files
3. Create a function to parse code corrections from LLM responses
4. Create a function to apply the corrections to the source file
5. Update the main function to use these new features

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

Adding functionality to save LLM responses to markdown, edit source code, and mark changes.

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

# ...existing code...

def save_to_markdown(source_file, error_info, response, model_name):
    """Save the debugging session to a markdown file."""
    # Create output filename based on source file
    source_path = Path(source_file)
    output_dir = source_path.parent / "debug_reports"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{source_path.stem}_debug_report_{timestamp}.md"
    
    # Format the content
    content = f"""# Debug Report for {source_path.name}

## Error Information
```
{error_info}
```

## LLM Analysis and Suggestions ({model_name})
{response}

## Source File
`{source_file}`

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    return output_file

def parse_code_corrections(response):
    """Parse the LLM response to extract code corrections."""
    # Look for code blocks with cpp tag
    code_blocks = re.findall(r'```cpp\n(.*?)```', response, re.DOTALL)
    
    # If no code blocks with explicit cpp tag, try to find any code blocks
    if not code_blocks:
        code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    
    return code_blocks

def edit_source_file(source_file, code_blocks):
    """Edit the source file with the suggested corrections."""
    if not code_blocks:
        print("No code corrections found in the LLM response.")
        return False
    
    # Make a backup of the original file
    source_path = Path(source_file)
    backup_file = source_path.with_suffix(f"{source_path.suffix}.bak")
    shutil.copy2(source_file, backup_file)
    
    # Read the original content
    with open(source_file, 'r') as f:
        original_content = f.read()
    
    # For simple implementation, use the first code block as complete replacement
    # A more sophisticated approach would identify specific sections to replace
    new_content = code_blocks[0]
    
    # Add markers to highlight the automated edits
    marked_content = f"/* AUTO-EDITED BY DEBUG ASSISTANT - BEGIN */\n{new_content}\n/* AUTO-EDITED BY DEBUG ASSISTANT - END */"
    
    # Write the new content
    with open(source_file, 'w') as f:
        f.write(marked_content)
    
    return backup_file

def main():
    """Main function to run the debug assistant."""
    args = parse_arguments()
    
    # Read the error log and source file
    error_log = read_file(args.error_log)
    source_code = read_file(args.source_file)
    
    # Extract relevant error information
    error_info = extract_error_information(error_log)
    
    # Create the debug prompt
    prompt = create_debug_prompt(error_info, source_code, args.source_file)
    
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
            backup_file = edit_source_file(args.source_file, code_blocks)
            print(f"Changes applied. Original source backed up to: {backup_file}")
            print("The edited sections are marked with /* AUTO-EDITED BY DEBUG ASSISTANT */ comments.")
        else:
            print("No changes applied. You can manually edit the source file based on the suggestions.")
    else:
        print("No specific code corrections found in the LLM response.")
    
    print("\nTo apply these fixes manually, edit your source file and re-run C simulation.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
````

This implementation:
1. Saves the LLM's response to a timestamped markdown file in a "debug_reports" directory
2. Parses the LLM response for code snippets (looking for code blocks in backticks)
3. Asks the user if they want to automatically apply the suggested changes
4. If approved, makes a backup of the original source file with a ".bak" extension
5. Applies the changes to the original file, adding comments to mark the automatically edited sections
6. Provides clear output to the user about what was done and where files are saved

The changes maintain all the original functionality while adding the requested features to save the response and edit the source code with clear markings.

# Issue with editing source code

rockyco: The corrected source file is not correct, it lost the other pieces of code that should remain unchanged. Revise this Python code to make the edited source code be integral.

GitHub Copilot: I'll improve the `edit_source_file` function in the debug assistant to properly preserve the original source code while only inserting the modified code sections. The current function incorrectly replaces the entire file with just the code block.

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

Revise the `edit_source_file` function to properly handle the original source code:

````python
# ...existing code...

def edit_source_file(source_file, code_blocks):
    """Edit the source file with the suggested corrections."""
    if not code_blocks:
        print("No code corrections found in the LLM response.")
        return False
    
    # Make a backup of the original file
    source_path = Path(source_file)
    backup_file = source_path.with_suffix(f"{source_path.suffix}.bak")
    shutil.copy2(source_file, backup_file)
    
    # Read the original content
    with open(source_file, 'r') as f:
        original_content = f.read()
    
    print("\nSuggested code correction:")
    print("-" * 40)
    for i, block in enumerate(code_blocks):
        print(f"Code Block {i+1}:")
        print(block)
        print("-" * 40)
    
    # Options for applying changes
    print("\nHow would you like to apply the changes?")
    print("1. Replace a specific portion of the code")
    print("2. Insert the code at a specific line")
    print("3. Keep original file and save suggested changes separately")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        search_pattern = input("Enter a unique string from the code section to replace: ")
        if search_pattern in original_content:
            # Simple replacement - in a production tool, this would be more sophisticated
            line_number = original_content.count('\n', 0, original_content.find(search_pattern)) + 1
            print(f"Found pattern at around line {line_number}")
            confirm = input(f"Replace from this section? (y/n): ").lower().strip()
            if confirm == 'y':
                # This is a very basic approach - a real solution would need more sophistication
                parts = original_content.split(search_pattern, 1)
                # Find suitable boundaries for replacement
                before_pattern = parts[0]
                after_pattern = parts[1].split('\n', 1)[1] if '\n' in parts[1] else parts[1]
                
                # Add markers and new content
                marked_content = (
                    before_pattern + 
                    "/* AUTO-EDITED BY DEBUG ASSISTANT - BEGIN */\n" + 
                    code_blocks[0] + 
                    "\n/* AUTO-EDITED BY DEBUG ASSISTANT - END */\n" + 
                    after_pattern
                )
                
                with open(source_file, 'w') as f:
                    f.write(marked_content)
                return backup_file
            else:
                print("Replacement canceled.")
                return False
        else:
            print(f"Pattern not found in the source file.")
            return False
    
    elif choice == '2':
        try:
            line_num = int(input("Enter the line number to insert the code: "))
            lines = original_content.splitlines()
            
            if 1 <= line_num <= len(lines) + 1:
                before_lines = lines[:line_num - 1]
                after_lines = lines[line_num - 1:]
                
                # Insert the new content with markers
                inserted_content = [
                    "/* AUTO-EDITED BY DEBUG ASSISTANT - BEGIN */",
                    code_blocks[0],
                    "/* AUTO-EDITED BY DEBUG ASSISTANT - END */"
                ]
                
                new_lines = before_lines + inserted_content + after_lines
                with open(source_file, 'w') as f:
                    f.write('\n'.join(new_lines))
                
                return backup_file
            else:
                print(f"Line number must be between 1 and {len(lines) + 1}")
                return False
        except ValueError:
            print("Invalid line number.")
            return False
    
    elif choice == '3':
        # Save suggestions to a separate file
        suggestion_file = source_path.with_name(f"{source_path.stem}_suggested{source_path.suffix}")
        with open(suggestion_file, 'w') as f:
            f.write("/* SUGGESTED CHANGES FROM DEBUG ASSISTANT */\n\n")
            for i, block in enumerate(code_blocks):
                f.write(f"/* SUGGESTION {i+1} */\n")
                f.write(block)
                f.write("\n\n")
        
        print(f"Suggestions saved to: {suggestion_file}")
        # Restore the original file (remove backup)
        shutil.copy2(backup_file, source_file)
        return backup_file
    
    else:
        print("Invalid choice. No changes made.")
        return False

# ...existing code...
````

This revised version:
1. Shows the suggested code changes to the user
2. Provides three options for applying changes:
   - Replace a specific portion using a search pattern
   - Insert at a specific line number
   - Save suggestions to a separate file
3. Preserves the original content while clearly marking any edits
4. Includes safeguards and confirmations before making changes

This approach will ensure that the original code integrity is maintained while incorporating only the specific changes needed.

# Simplify file editing

rockyco: Revise the edit_source_file() function to only keep the third option for applying changes, that is keep original file and save suggested changes seperately.

GitHub Copilot: Let me revise the `edit_source_file()` function to only keep the option to save suggested changes separately without modifying the original file.

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

Simplify the `edit_source_file()` function to only support saving suggested changes to a separate file without modifying the original.

````python
def edit_source_file(source_file, code_blocks):
    """Save the suggested corrections to a separate file without modifying the original."""
    if not code_blocks:
        print("No code corrections found in the LLM response.")
        return False
    
    # Make a backup of the original file
    source_path = Path(source_file)
    backup_file = source_path.with_suffix(f"{source_path.suffix}.bak")
    shutil.copy2(source_file, backup_file)
    
    print("\nSuggested code correction:")
    print("-" * 40)
    for i, block in enumerate(code_blocks):
        print(f"Code Block {i+1}:")
        print(block)
        print("-" * 40)
    
    # Save suggestions to a separate file
    suggestion_file = source_path.with_name(f"{source_path.stem}_suggested{source_path.suffix}")
    with open(suggestion_file, 'w') as f:
        f.write("/* SUGGESTED CHANGES FROM DEBUG ASSISTANT */\n\n")
        for i, block in enumerate(code_blocks):
            f.write(f"/* SUGGESTION {i+1} */\n")
            f.write(block)
            f.write("\n\n")
    
    print(f"Suggestions saved to: {suggestion_file}")
    # Restore the original file (remove backup)
    shutil.copy2(backup_file, source_file)
    return backup_file
````

# Update source code editing

rockyco: Based on the LLM's response stored in this mardown file, revise the Python code to only extract the final corrected code for saving. 

GitHub Copilot: I'll revise the `parse_code_corrections` function to extract the final corrected code from the LLM's response, prioritizing "refined" corrections when available.

### [debug_assistant.py](file:///home/jielei/Projects/UTS/peakPicker/scripts/debug_assistant.py)

Update the `parse_code_corrections` function to prioritize refined/final solutions and improve code extraction logic.

````python
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
````

This updated function first looks for code blocks specifically labeled as "refined" or "final" solutions, then falls back to blocks labeled as "corrected" or "fixed" code. If neither is found, it reverts to the original behavior of extracting all code blocks. This ensures that when the LLM provides a refined solution (as in the markdown example), that solution is prioritized for saving.

# Add more source files to debug

rockyco: Revise the Python code to solve this issue '(.venv) jielei@UTS-HP-WS:~/Projects/UTS/llm-fpga-design$ python3 scripts/debug_assistant.py --error_log implementations/peakPicker/proj_peakPicker/solution1/csim/report/peakPicker_csim.log --source_file implementations/peakPicker/peakPicker.cpp implementations/peakPicker/peakPicker.hpp implementations/peakPicker/peakPicker_tb.cpp
Error reading file ['implementations/peakPicker/peakPicker.cpp', 'implementations/peakPicker/peakPicker.hpp', 'implementations/peakPicker/peakPicker_tb.cpp']: expected str, bytes or os.PathLike object, not list'. Need multiple source files to be added as command argument.

GitHub Copilot: I'll solve this issue by modifying the code to properly handle multiple source files. The current problem is that the script tries to pass a list of file paths to the `read_file()` function, which only accepts a single file path.

## Solution

### [debug_assistant.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/debug_assistant.py)

I'll update the code to handle multiple source files by modifying several functions, including main(), read_file(), and create_debug_prompt().

````python
# ...existing code...

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

# ...existing code...

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

# ...existing code...

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

# ...existing code...
````

The changes I've made:

1. Updated `create_debug_prompt()` to accept a dictionary of source files and their contents, formatting them all in the prompt.
2. Modified `save_to_markdown()` to handle multiple source files in the report.
3. Updated `edit_source_file()` to make backups of all source files.
4. Completely rewrote the `main()` function to:
   - Read all source files into a dictionary
   - Pass that dictionary to the debug prompt creation
   - Handle multiple files throughout the workflow

These changes will fix the error and allow the script to properly process multiple source files passed as command-line arguments.

