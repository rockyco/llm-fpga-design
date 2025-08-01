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

def categorize_error(error_info):
    """Categorize the error for better documentation"""
    categories = {
        "memory": ["out of bounds", "segmentation fault", "memory access", "buffer overflow", "invalid pointer"],
        "datatype": ["incompatible types", "cannot convert", "invalid conversion", "type mismatch"],
        "syntax": ["expected", "missing", "undeclared", "not declared", "syntax error", "before"],
        "interface": ["interface mismatch", "port", "incompatible interface", "input port", "output port"],
        "simulation": ["simulation failed", "csim failed", "test bench", "verification failed", "result mismatch"],
        "pragma": ["pragma", "directive", "unroll", "pipeline", "dataflow", "array_partition"],
        "latency": ["timing", "latency", "cannot achieve", "II constraint"],
        "resource": ["insufficient resources", "DSP", "BRAM", "LUT", "FF", "resource"]
    }
    
    result = {"primary_category": "unknown", "all_categories": [], "details": {}}
    
    error_lower = error_info.lower()
    
    # Check each category
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in error_lower:
                result["all_categories"].append(category)
                # Only record details for categories we found
                if category not in result["details"]:
                    result["details"][category] = []
                # Extract the specific line containing this keyword
                for line in error_info.splitlines():
                    if keyword in line.lower():
                        if line not in result["details"][category]:
                            result["details"][category].append(line)
    
    # Determine primary category (the one with most matches or first found)
    if result["all_categories"]:
        category_counts = {}
        for category in result["all_categories"]:
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        # Get category with highest count
        result["primary_category"] = max(category_counts.items(), key=lambda x: x[1])[0]
    
    return result

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
    """Save the debugging session to a markdown file with enhanced error categorization."""
    # Create output filename based on first source file
    source_path = Path(source_files[0])
    output_dir = source_path.parent / "debug_reports"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{source_path.stem}_debug_report_{timestamp}.md"
    
    # Format the content
    files_list = "\n".join([f"- `{f}`" for f in source_files])
    
    # Categorize the error
    error_analysis = categorize_error(error_info)
    primary_category = error_analysis["primary_category"].capitalize()
    all_categories = ", ".join(error_analysis["all_categories"]).capitalize() if error_analysis["all_categories"] else "Unknown"
    
    # Extract bug/fix summary for documentation
    bug_pattern = r"(?:The main bug|The error|The issue|The problem).*?(?=\n\n|\n##|\Z)"
    bug_match = re.search(bug_pattern, response, re.DOTALL | re.IGNORECASE)
    bug_summary = bug_match.group(0) if bug_match else "No concise bug description found."
    
    fix_pattern = r"(?:The fix|The solution|To fix this|The correction).*?(?=\n\n|\n##|\Z)"
    fix_match = re.search(fix_pattern, response, re.DOTALL | re.IGNORECASE)
    fix_summary = fix_match.group(0) if fix_match else "No concise fix description found."
    
    # Create detailed error section
    error_details = ""
    for category, lines in error_analysis["details"].items():
        error_details += f"### {category.capitalize()} Issues\n"
        for line in lines:
            error_details += f"- {line}\n"
        error_details += "\n"
    
    content = f"""# Debug Report

## Error Summary
- **Primary Error Category**: {primary_category}
- **All Categories Detected**: {all_categories}
- **Timestamp**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Bug Summary
{bug_summary}

## Fix Summary
{fix_summary}

## Full Error Information
```
{error_info}
```

{error_details}

## LLM Analysis and Suggestions ({model_name})
{response}

## Source Files
{files_list}
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    # Also save a JSON file with structured error information for better machine readability
    json_file = output_dir / f"{source_path.stem}_debug_data_{timestamp}.json"
    error_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "files": source_files,
        "error_analysis": error_analysis,
        "bug_summary": bug_summary,
        "fix_summary": fix_summary,
        "model_used": model_name
    }
    
    with open(json_file, 'w') as f:
        json.dump(error_data, f, indent=2)
    
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
