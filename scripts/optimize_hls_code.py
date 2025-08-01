#!/usr/bin/env python3

import argparse
import os
import sys
import requests
import json
import re
import openai
import google.generativeai as genai
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize HLS C++ code for better performance using LLM')
    parser.add_argument('--source_dir', required=True, 
                        help='Directory containing HLS source files to optimize')
    parser.add_argument('--prompt', required=False, 
                        help='Path to prompt template file (or prompt name)')
    parser.add_argument('--output_dir', default=None, 
                        help='Directory to save optimized HLS code (defaults to source_dir)')
    parser.add_argument('--model', default='gemini-2.5-pro-exp-03-25', 
                        help='LLM model to use')
    parser.add_argument('--primary_goal', required=False, default="Reduce latency", 
                        help='Primary optimization goal (e.g., "Reduce latency by 30%")')
    parser.add_argument('--secondary_goal', required=False, default="Maintain resource usage", 
                        help='Secondary optimization goal (e.g., "Maintain resource usage")')
    parser.add_argument('--api_key', 
                        help='API key for LLM service')
    return parser.parse_args()

def read_file(file_path):
    """Read and return the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def find_source_files(source_dir):
    """Find all relevant HLS source files in the directory."""
    source_files = {}
    for ext in ['.cpp', '.hpp', '.h', '_tb.cpp']:
        for file in Path(source_dir).glob(f'*{ext}'):
            source_files[file.name] = str(file)
    
    # Also look for csynth.rpt or other report files
    for report_file in Path(source_dir).glob('**/csynth.rpt'):
        source_files['csynth.rpt'] = str(report_file)
    
    # Look for implementation reports
    for report_file in Path(source_dir).glob('**/verilog/report/**/*.rpt'):
        source_files[f'report_{report_file.name}'] = str(report_file)
    
    return source_files

def extract_performance_metrics(source_dir):
    """Extract performance metrics from synthesis and implementation reports."""
    metrics = {}
    
    # Look for csynth.rpt
    csynth_path = None
    for path in Path(source_dir).glob('**/csynth.rpt'):
        csynth_path = path
        break
    
    if csynth_path:
        try:
            csynth_content = read_file(str(csynth_path))
            
            # Extract latency information
            latency_match = re.search(r'Latency \(cycles\)\s*\|\s*min\s*\|\s*max\s*\|\s*min/max\s*\|\s*\n\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*\n\s*\|\s*(\d+)\s*\|\s*(\d+)', csynth_content)
            if latency_match:
                metrics['latency_min'] = int(latency_match.group(1))
                metrics['latency_max'] = int(latency_match.group(2))
            
            # Extract resource utilization
            resource_pattern = r'(\|\s*([A-Za-z0-9]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+%?\s*)\|)'
            resource_matches = re.findall(resource_pattern, csynth_content)
            
            if resource_matches:
                metrics['resources'] = {}
                for match in resource_matches:
                    resource_type = match[1].strip()
                    used = int(match[2])
                    total = int(match[3]) if match[3] != '0' else 0
                    metrics['resources'][resource_type] = {
                        'used': used,
                        'total': total,
                        'utilization': f"{(used/total*100):.2f}%" if total > 0 else "N/A"
                    }
        except Exception as e:
            print(f"Error extracting metrics from csynth.rpt: {e}")
    
    # Format metrics as a string for the prompt
    metrics_str = "## Performance Metrics\n\n"
    
    if 'latency_min' in metrics:
        metrics_str += f"### Latency\n- Minimum: {metrics['latency_min']} cycles\n- Maximum: {metrics['latency_max']} cycles\n\n"
    
    if 'resources' in metrics:
        metrics_str += "### Resource Utilization\n"
        for resource, values in metrics['resources'].items():
            metrics_str += f"- {resource}: {values['used']} used / {values['total']} total ({values['utilization']})\n"
    
    if metrics_str == "## Performance Metrics\n\n":
        metrics_str += "No performance metrics available from synthesis reports."
    
    return metrics_str, metrics

def create_optimization_prompt(source_files, performance_metrics, prompt_template, primary_goal, secondary_goal):
    """Create the prompt for code optimization."""
    # Load source file contents
    source_contents = {}
    for name, path in source_files.items():
        if name.endswith(('.cpp', '.hpp', '.h', '_tb.cpp')):
            try:
                source_contents[name] = read_file(path)
            except:
                source_contents[name] = f"Error reading {path}"
    
    # Create source files section
    source_files_str = "## Source Files\n\n"
    for name, content in source_contents.items():
        source_files_str += f"### {name}\n```cpp\n{content}\n```\n\n"
    
    # Replace placeholders in template
    prompt = prompt_template
    replacements = {
        "SOURCE_FILES": source_files_str,
        "PERFORMANCE_METRICS": performance_metrics,
        "PRIMARY_GOAL": primary_goal,
        "SECONDARY_GOAL": secondary_goal
    }
    
    for key, value in replacements.items():
        if f"{{{{{key}}}}}" in prompt:
            prompt = prompt.replace(f"{{{{{key}}}}}", value)
    
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
            {"role": "system", "content": "You are an expert FPGA developer specializing in HLS C++ optimization."},
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

def query_claude(prompt, model="claude-3-sonnet-20240229"):
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
        "contents": [{"parts": [{"text": "You are an expert FPGA developer specializing in HLS C++ optimization.\n\n" + prompt}]}],
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

def query_llm(prompt, model="gemini-2.5-pro-exp-03-25"):
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

def extract_optimized_code(llm_response):
    """Extract optimized code blocks from LLM response."""
    optimized_code = {}
    
    # Extract all code blocks with filenames
    filename_patterns = [
        r'###\s+([a-zA-Z0-9_]+\.[ch]pp)\s*```cpp\s*(.*?)```',
        r'File:\s*([a-zA-Z0-9_]+\.[ch]pp)\s*```cpp\s*(.*?)```',
        r'```cpp\s*//\s*([a-zA-Z0-9_]+\.[ch]pp)\s*(.*?)```'
    ]
    
    for pattern in filename_patterns:
        matches = re.findall(pattern, llm_response, re.DOTALL)
        for filename, code in matches:
            optimized_code[filename.strip()] = code.strip()
    
    return optimized_code

def apply_optimizations(source_dir, output_dir, optimized_code, llm_response):
    """Apply the optimized code to the files."""
    # Create output directory if it doesn't exist
    if output_dir and output_dir != source_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = source_dir
        
        # Create backup directory
        backup_dir = os.path.join(source_dir, "backup_original")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup original files
        for filename in optimized_code.keys():
            source_file = os.path.join(source_dir, filename)
            if os.path.exists(source_file):
                backup_file = os.path.join(backup_dir, filename)
                try:
                    with open(source_file, 'r') as src, open(backup_file, 'w') as dst:
                        dst.write(src.read())
                    print(f"Backed up {filename} to {backup_dir}/")
                except Exception as e:
                    print(f"Error backing up {filename}: {e}")
    
    # Write optimized code to files
    for filename, code in optimized_code.items():
        output_file = os.path.join(output_dir, filename)
        try:
            with open(output_file, 'w') as f:
                f.write(code)
            print(f"Applied optimization to {filename}")
        except Exception as e:
            print(f"Error writing optimized code to {filename}: {e}")
    
    # Create a log file with the optimization details
    log_file = os.path.join(output_dir, "optimization_log.md")
    with open(log_file, 'w') as f:
        f.write("# HLS Code Optimization Log\n\n")
        f.write(f"Optimization performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Files Modified\n\n")
        for filename in optimized_code.keys():
            f.write(f"- {filename}\n")
        f.write("\n## Optimizations Applied\n\n")
        f.write(llm_response)
    
    print(f"Optimization log saved to {log_file}")
    
    return log_file

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
    
    # Find source files
    source_files = find_source_files(args.source_dir)
    if not source_files:
        print(f"Error: No source files found in {args.source_dir}")
        sys.exit(1)
    
    print(f"Found {len(source_files)} source files in {args.source_dir}")
    for name in source_files:
        print(f"  - {name}")
    
    # Extract performance metrics
    performance_metrics_str, metrics = extract_performance_metrics(args.source_dir)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.source_dir
    
    # Load prompt template
    if args.prompt and os.path.isfile(args.prompt):
        prompt_template = read_file(args.prompt)
    else:
        # Try to find the performance_optimization.md prompt
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        prompt_path = os.path.join(project_dir, "prompts", "performance_optimization.md")
        
        if os.path.isfile(prompt_path):
            prompt_template = read_file(prompt_path)
        else:
            print("Error: Could not find performance_optimization.md prompt")
            basic_template = """# HLS Performance Optimization

Please analyze the provided HLS source code and suggest optimizations to improve performance based on the provided metrics.

## Source Files
{{SOURCE_FILES}}

## Performance Metrics
{{PERFORMANCE_METRICS}}

## Optimization Goals
- Primary goal: {{PRIMARY_GOAL}}
- Secondary goal: {{SECONDARY_GOAL}}

Please suggest specific HLS pragmas and code modifications to achieve these goals.
"""
            prompt_template = basic_template
            print("Using basic optimization prompt template")
    
    # Create the optimization prompt
    optimization_prompt = create_optimization_prompt(
        source_files, 
        performance_metrics_str, 
        prompt_template, 
        args.primary_goal, 
        args.secondary_goal
    )
    
    # Get component name from directory
    component_name = os.path.basename(os.path.normpath(args.source_dir))
    
    print(f"Generating optimizations for {component_name}...")
    print(f"Using model: {args.model}")
    print(f"Primary goal: {args.primary_goal}")
    print(f"Secondary goal: {args.secondary_goal}")
    
    # Call the LLM
    llm_response = query_llm(optimization_prompt, args.model)
    
    # Extract optimized code
    optimized_code = extract_optimized_code(llm_response)
    
    if not optimized_code:
        print("Warning: No optimized code blocks detected in the LLM response.")
        # Save the full response
        response_path = os.path.join(args.source_dir, "optimization_suggestions.md") 
        with open(response_path, 'w') as f:
            f.write(llm_response)
        print(f"Saved optimization suggestions to {response_path}")
        sys.exit(1)
    
    # Apply optimizations
    log_file = apply_optimizations(args.source_dir, output_dir, optimized_code, llm_response)
    
    print("\nOptimization complete!")
    print(f"Optimized {len(optimized_code)} files")
    print(f"Output directory: {output_dir}")
    print(f"Optimization log: {log_file}")
    
    # Track feedback for prompt improvement
    feedback_file = os.path.join(project_dir, "prompt_feedback.json")
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    with open(feedback_file, "a") as f:
        feedback = {
            "prompt": "performance_optimization",
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "component": component_name,
            "status": "success" if optimized_code else "partial",
            "files_optimized": len(optimized_code),
            "optimized_files": list(optimized_code.keys())
        }
        f.write(json.dumps(feedback) + "\n")
    
    # Return the output directory for the orchestrator
    print(output_dir)

if __name__ == "__main__":
    main()
