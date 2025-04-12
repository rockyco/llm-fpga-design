#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import requests
import re
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

def update_prompt_with_feedback(format_type, generation_result, template_dir=None):
    """Update prompt template with feedback from generation results."""
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
    
    # Check if template exists
    if not os.path.exists(template_file):
        logger.warning(f"Cannot update template - file not found: {template_file}")
        return False
        
    try:
        # Read existing template
        with open(template_file, 'r') as f:
            template_content = f.read()
            
        # Create backup of original
        backup_dir = os.path.join(template_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"{os.path.basename(template_file)}.{timestamp}")
        
        with open(backup_file, 'w') as f:
            f.write(template_content)
        
        # Determine if the generation was successful
        if "error" in generation_result:
            # Add error information to template
            error_info = generation_result["error"]
            
            # Check if we already have a COMMON ERRORS section
            if "## COMMON ERRORS" in template_content:
                # Add to existing section
                parts = template_content.split("## COMMON ERRORS")
                updated_content = parts[0] + "## COMMON ERRORS" + parts[1].rstrip() + f"\n- {error_info}\n"
            else:
                # Add new section
                updated_content = template_content.rstrip() + f"\n\n## COMMON ERRORS\n\n- {error_info}\n"
                
            # Write updated template
            with open(template_file, 'w') as f:
                f.write(updated_content)
                
            logger.info(f"Updated template {template_file} with error information")
            return True
        else:
            # For successful generation, identify what might have contributed to success
            # This is simplified - a real implementation would analyze the content
            
            # Check if we have BEST PRACTICES section
            if "## BEST PRACTICES" in template_content:
                # Add to existing section
                now = datetime.datetime.now().strftime("%Y-%m-%d")
                parts = template_content.split("## BEST PRACTICES")
                updated_content = parts[0] + "## BEST PRACTICES" + parts[1].rstrip() + f"\n- Successfully generated documentation on {now}\n"
            else:
                # Add new section
                now = datetime.datetime.now().strftime("%Y-%m-%d")
                updated_content = template_content.rstrip() + f"\n\n## BEST PRACTICES\n\n- Successfully generated documentation on {now}\n"
                
            # Write updated template
            with open(template_file, 'w') as f:
                f.write(updated_content)
                
            logger.info(f"Updated template {template_file} with success information")
            return True
    
    except Exception as e:
        logger.error(f"Error updating template {template_file}: {e}")
        return False

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

# New function to extract performance data for visualization
def extract_performance_data(metrics):
    """Extract and format performance data for visualization."""
    data = {
        "resources": {},
        "timing": {},
        "latency": {}
    }
    
    # Extract resource usage data
    if "resources" in metrics and metrics["resources"]:
        for impl, resources in metrics["resources"].items():
            data["resources"][impl] = resources
    
    # Extract timing data
    if "timing" in metrics and metrics["timing"]:
        for impl, timing in metrics["timing"].items():
            # Format MHz values with two decimal places and suffix
            formatted_timing = {}
            for key, value in timing.items():
                if "_MHz" in key and isinstance(value, (int, float)):
                    formatted_timing[key] = f"{value:.2f} MHz"
                else:
                    formatted_timing[key] = value
            data["timing"][impl] = formatted_timing
    
    # Extract latency data
    if "latency" in metrics and metrics["latency"]:
        for impl, latency in metrics["latency"].items():
            data["latency"][impl] = latency
    
    return data

# Update the function to generate Mermaid diagram examples with proper escaping
def generate_diagram_examples():
    """Generate example Mermaid diagrams to include in prompts."""
    examples = {
        "flowchart": '''```mermaid
flowchart TD
    A["Input Data"] --> B["Pre-processing"]
    B --> C{"Decision"}
    C -->|"Option 1"| D["Result 1"]
    C -->|"Option 2"| E["Result 2"]
```''',
        "sequence": '''```mermaid
sequenceDiagram
    participant Host
    participant FPGA
    Host->>FPGA: "Send Input Data"
    FPGA->>FPGA: "Process Data"
    FPGA->>Host: "Return Results"
```''',
        "gantt": '''```mermaid
gantt
    title Pipeline Execution Timeline
    dateFormat  s
    axisFormat %S
    section Pipeline
    Stage 1      :a1, 0, 2s
    Stage 2      :a2, after a1, 3s
    Stage 3      :a3, after a2, 1s
```''',
        "class": '''```mermaid
classDiagram
    class Component {
        +input_ports
        +output_ports
        +process()
    }
    class Submodule {
        +calculate()
    }
    Component --> Submodule
```''',
        "state": '''```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: "start"
    Processing --> Idle: "done"
    Processing --> Error: "error"
    Error --> Idle: "reset"
```'''
    }
    
    return examples

# New function to generate chart examples for performance data
def generate_chart_examples():
    """Generate examples of charts for displaying performance data."""
    examples = {
        "resource_table": '''| Resource | Implementation 1 | Implementation 2 |
|----------|-----------------|-----------------|
| LUT      | 1500            | 1200            |
| FF       | 2000            | 1800            |
| DSP      | 5               | 4               |
| BRAM     | 2               | 2               |''',
        
        "timing_table": '''| Timing      | Value (ns) | Frequency (MHz) |
|-------------|-----------|----------------|
| Target      | 10.0      | 100.0          |
| Achieved    | 8.5       | 117.6          |
| Slack       | 1.5       | -              |''',
        
        "latency_table": '''| Test Case | Latency (cycles) | Throughput (samples/cycle) |
|-----------|-----------------|----------------------------|
| Case 1    | 100             | 1.0                        |
| Case 2    | 120             | 0.8                        |'''
    }
    
    return examples

def get_debug_prompt_template():
    """Load a debug prompt template or return a default one."""
    # Try to find template in standard locations
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    template_file = os.path.join(project_dir, "prompts", "debug_template.md")
    
    # Check if template exists, otherwise return a default template
    if os.path.exists(template_file):
        with open(template_file, 'r') as f:
            return f.read()
    
    # Default debug template if file not found
    return """# Debug Assistance Request

## Error Log
{{ERROR_LOG}}

## Source Files
{{SOURCE_FILES}}

Please analyze the error log and source files to identify the issue.
Provide:
1. Root cause of the error
2. Suggested fix with code example
3. Explanation of the fix
"""

def extract_design_insights(prompt):
    """Extract design insights from the prompt to enhance documentation."""
    insights = {
        "algorithm_insights": [],
        "design_decisions": [],
        "implementation_challenges": [],
        "bugs": [],
        "fixes": [],
        "optimization_strategies": []
    }
    
    # Extract LLM Design Insights section
    llm_insights_pattern = r'## LLM Design Insights\s*\n(.*?)(?=\n##|\Z)'
    llm_insights_match = re.search(llm_insights_pattern, prompt, re.DOTALL)
    
    if llm_insights_match:
        llm_insights_content = llm_insights_match.group(1)
        
        # Process algorithm insights
        algo_pattern = r'### Algorithm Insights\s*\n(.*?)(?=\n###|\Z)'
        algo_match = re.search(algo_pattern, llm_insights_content, re.DOTALL)
        if algo_match:
            insights["algorithm_insights"] = re.findall(r'- (.*?)(?=\n-|\n\n|\Z)', algo_match.group(1), re.DOTALL)
        
        # Process design decisions
        design_pattern = r'### Design Decisions\s*\n(.*?)(?=\n###|\Z)'
        design_match = re.search(design_pattern, llm_insights_content, re.DOTALL)
        if design_match:
            insights["design_decisions"] = re.findall(r'- (.*?)(?=\n-|\n\n|\Z)', design_match.group(1), re.DOTALL)
        
        # Process implementation challenges
        challenge_pattern = r'### Implementation Challenges\s*\n(.*?)(?=\n###|\Z)'
        challenge_match = re.search(challenge_pattern, llm_insights_content, re.DOTALL)
        if challenge_match:
            insights["implementation_challenges"] = re.findall(r'- (.*?)(?=\n-|\n\n|\Z)', challenge_match.group(1), re.DOTALL)
        
        # Process bugs
        bugs_pattern = r'### Bugs Identified\s*\n(.*?)(?=\n###|\Z)'
        bugs_match = re.search(bugs_pattern, llm_insights_content, re.DOTALL)
        if bugs_match:
            insights["bugs"] = re.findall(r'- (.*?)(?=\n-|\n\n|\Z)', bugs_match.group(1), re.DOTALL)
        
        # Process fixes
        fixes_pattern = r'### Applied Fixes\s*\n(.*?)(?=\n###|\Z)'
        fixes_match = re.search(fixes_pattern, llm_insights_content, re.DOTALL)
        if fixes_match:
            insights["fixes"] = re.findall(r'- (.*?)(?=\n-|\n\n|\Z)', fixes_match.group(1), re.DOTALL)
        
        # Process optimization strategies
        opt_pattern = r'### Optimization Strategies\s*\n(.*?)(?=\n###|\Z)'
        opt_match = re.search(opt_pattern, llm_insights_content, re.DOTALL)
        if opt_match:
            insights["optimization_strategies"] = re.findall(r'- (.*?)(?=\n-|\n\n|\Z)', opt_match.group(1), re.DOTALL)
    
    return insights

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
    
    # Categorize error for better prompting
    error_analysis = categorize_error(error_info)
    primary_category = error_analysis.get("primary_category", "unknown")
    error_categories = error_analysis.get("all_categories", [])
    
    # Add error categorization to help the LLM focus
    prompt += f"\n## Error Categorization\n"
    prompt += f"Primary error category: {primary_category}\n"
    if error_categories:
        prompt += f"All categories detected: {', '.join(error_categories)}\n\n"
    
    # Add specific guidance based on error category
    if primary_category == "memory":
        prompt += "Focus on memory access patterns, array bounds, and pointer operations.\n"
    elif primary_category == "datatype":
        prompt += "Focus on data type conversions, bitwidth issues, and type compatibility.\n"
    elif primary_category == "syntax":
        prompt += "Focus on C++ syntax requirements and HLS-specific syntax limitations.\n"
    elif primary_category == "interface":
        prompt += "Focus on interface definitions, port mappings, and AXI/streaming protocols.\n"
    
    return prompt

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
        
        # Extract design insights from the prompt
        design_insights = extract_design_insights(base_prompt)
        logger.info(f"Extracted {sum(len(v) for v in design_insights.values())} design insights from LLM responses")
        
        # Extract performance metrics for visualization if available
        performance_data = {}
        try:
            # Look for metrics section in the prompt
            metrics_match = re.search(r"## Performance Metrics\s*\n(.*?)(?:\n##|\Z)", base_prompt, re.DOTALL)
            if metrics_match:
                metrics_text = metrics_match.group(1)
                # Try to parse metrics into a dictionary (simplified version)
                metrics = {}
                for section in ["resources", "timing", "latency"]:
                    section_match = re.search(f"### {section.capitalize()}.*?\n(.*?)(?:\n###|\Z)", metrics_text, re.DOTALL, re.IGNORECASE)
                    if section_match:
                        metrics[section] = {}
                        lines = section_match.group(1).strip().split('\n')
                        for line in lines:
                            if line.startswith('-') and ':' in line:
                                parts = line.strip('- ').split(':')
                                if len(parts) >= 2:
                                    impl = parts[0].strip()
                                    metrics[section][impl] = {}
                                    values = ':'.join(parts[1:]).split(',')
                                    for val in values:
                                        if ':' in val:
                                            k, v = val.split(':', 1)
                                            metrics[section][impl][k.strip()] = v.strip()
                
                # Now extract structured performance data for visualization
                if metrics:
                    performance_data = extract_performance_data(metrics)
                    
                    # Format metrics for inclusion in documentation
                    formatted_metrics = format_metrics_as_text(metrics)
                    logger.info(f"Formatted performance metrics for documentation")
                    
                    # Check if performance metrics sections exist in base_prompt
                    metrics_section_match = re.search(r"## Performance Metrics\s*\n", base_prompt)
                    
                    if metrics_section_match:
                        # If Performance Metrics section exists, replace its subsections
                        # Find where the section starts
                        section_start = metrics_section_match.end()
                        
                        # Find where the section ends (next ## heading or end of text)
                        next_section_match = re.search(r"\n##\s", base_prompt[section_start:])
                        section_end = section_start + (next_section_match.start() if next_section_match else len(base_prompt[section_start:]))
                        
                        # Replace the content of the Performance Metrics section
                        base_prompt = base_prompt[:section_start] + formatted_metrics + base_prompt[section_end:]
                    else:
                        # No Performance Metrics section exists, append it
                        base_prompt += f"\n\n## Performance Metrics\n{formatted_metrics}"
                        
                    logger.info(f"Extracted performance data for visualization: {len(performance_data)} categories")
        except Exception as e:
            logger.warning(f"Error extracting performance data: {e}")
        
        results = {}
        
        # Generate each requested format
        for fmt in formats:
            logger.info(f"Generating {fmt} documentation")
            
            # Try to load template for this format
            template = load_prompt_template(fmt, template_dir)
            
            if template:
                # Combine base prompt with template
                format_prompt = template
                
                # Extract data from base_prompt to fill template
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
                    
                    # Add formatted performance data for visualization
                    if performance_data:
                        data["performance_data"] = json.dumps(performance_data)
                    
                    # Extract errors
                    errors_match = re.search(r"### Errors Encountered\n(.*?)(?:\n###|\Z)", base_prompt, re.DOTALL)
                    if errors_match:
                        data["errors_encountered"] = errors_match.group(1).strip()
                    
                    # Extract debugging methods
                    debug_match = re.search(r"### Debugging Methods\n(.*?)(?:\n##|\Z)", base_prompt, re.DOTALL)
                    if debug_match:
                        data["debugging_methods"] = debug_match.group(1).strip()
                    
                    # Add visualization examples to the data
                    data["diagram_examples"] = json.dumps(generate_diagram_examples())
                    data["chart_examples"] = json.dumps(generate_chart_examples())
                    
                    # Add LLM design insights to the data
                    data["algorithm_insights"] = json.dumps(design_insights["algorithm_insights"])
                    data["design_decisions"] = json.dumps(design_insights["design_decisions"])
                    data["implementation_challenges"] = json.dumps(design_insights["implementation_challenges"])
                    data["bugs_identified"] = json.dumps(design_insights["bugs"])
                    data["applied_fixes"] = json.dumps(design_insights["fixes"])
                    data["optimization_strategies"] = json.dumps(design_insights["optimization_strategies"])
                    
                    # Replace template placeholders with extracted data
                    for key, value in data.items():
                        format_prompt = format_prompt.replace(f"{{{key}}}", value)
                    
                except Exception as e:
                    logger.warning(f"Error parsing base prompt: {e}")
                    # Fall back to basic prompt + template approach
                    format_prompt = f"{base_prompt}\n\nPlease generate ONLY the {fmt} document now, using the template guidelines below:\n\n{template}"
                    
                    # Add visualization instructions even in fallback case
                    format_prompt += "\n\n## Visualization Requirements\n"
                    format_prompt += "Please include appropriate diagrams using Mermaid notation for architecture and data flow.\n"
                    format_prompt += "Use tables to present performance metrics and comparative analysis.\n"
                    format_prompt += "For diagrams, use ```mermaid blocks in the markdown.\n"
            else:
                # No template available, use base prompt
                format_prompt = f"{base_prompt}\n\nPlease generate ONLY the {fmt} document now."
                
                # Add basic visualization instructions
                format_prompt += "\n\n## Visualization Requirements\n"
                format_prompt += "Please include appropriate diagrams using Mermaid notation for architecture and data flow.\n"
                format_prompt += "Use tables to present performance metrics and comparative analysis.\n"
                format_prompt += "For diagrams, use ```mermaid blocks in the markdown.\n"
            
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
                # Update template with error feedback
                update_prompt_with_feedback(fmt, {"error": response.get("error", "Unknown error")}, template_dir)
                continue
            
            # Extract and save the content
            content = response.get("text", "")
            output_file = os.path.join(output_dir, f"{fmt}.md")
            
            with open(output_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Saved {fmt} documentation to {output_file}")
            results[fmt] = {"file": output_file, "size": len(content)}
            
            # Update template with success feedback
            update_prompt_with_feedback(fmt, {"success": True}, template_dir)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in documentation generation: {str(e)}")
        return {"error": str(e)}

# Function to categorize error - will also be used by generate_documentation.py
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

def main():
    parser = argparse.ArgumentParser(description="Generate documentation from LLM")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated files")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-exp-03-25", help="LLM model to use")
    parser.add_argument("--formats", nargs="+", default=["readme"], choices=["readme", "paper"], 
                       help="Documentation formats to generate (readme, paper)")
    parser.add_argument("--api_key", help="API key for LLM service (overrides environment variable)")
    parser.add_argument("--template_dir", help="Directory containing prompt templates")
    parser.add_argument("--include_diagrams", action="store_true", help="Include diagrams in documentation")
    parser.add_argument("--visualization_style", choices=["mermaid", "ascii", "both"], default="mermaid",
                       help="Style of visualizations to include")
    
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
    main()

def format_metrics_as_text(metrics):
    """Format metrics dictionary as readable text"""
    perf_text = ""
    
    # Format resource utilization
    if "resources" in metrics and metrics["resources"]:
        perf_text += "### Resource Utilization\n"
        for impl, resources in metrics["resources"].items():
            perf_text += f"- **{impl}**: "
            for res, val in resources.items():
                perf_text += f"{res}: {val}, "
            perf_text = perf_text.rstrip(", ") + "\n"
    
    # Format timing information - Enhanced to highlight importance
    if "timing" in metrics and metrics["timing"]:
        perf_text += "\n### Timing (CRITICAL PERFORMANCE METRIC)\n"
        perf_text += "The timing metrics below show clock period and frequency information, which are critical for understanding performance constraints and throughput capabilities.\n\n"
        
        # Create a table for better visualization in both README and paper
        perf_text += "| Implementation | Period (ns) | Frequency (MHz) | Timing Margin |\n"
        perf_text += "|----------------|------------|----------------|---------------|\n"
        
        for impl, timing in metrics["timing"].items():
            # Get target, post-synthesis and post-route timing info if available
            target = timing.get("Target", "-")
            if not isinstance(target, (int, float)):
                try:
                    target = float(target)
                except (ValueError, TypeError):
                    target = "-"
            
            post_synth = timing.get("Post-Synthesis", "-")
            if not isinstance(post_synth, (int, float)):
                try:
                    post_synth = float(post_synth)
                except (ValueError, TypeError):
                    post_synth = "-"
            
            post_route = timing.get("Post-Route", "-")
            if not isinstance(post_route, (int, float)):
                try:
                    post_route = float(post_route)
                except (ValueError, TypeError):
                    post_route = "-"
            
            # Calculate margin if possible
            margin = "-"
            if isinstance(target, (int, float)) and isinstance(post_route, (int, float)):
                margin = f"{target - post_route:.2f} ns"
            
            # Calculate frequency
            freq = "-"
            if isinstance(post_route, (int, float)) and post_route > 0:
                freq = f"{1000.0/post_route:.2f}"
            
            perf_text += f"| {impl} | {post_route if post_route != '-' else '-'} | {freq} | {margin} |\n"
        
        perf_text += "\nNote: Timing is a critical performance metric that affects maximum clock frequency.\n"
    
    # Format latency information - Enhanced to highlight importance
    if "latency" in metrics and metrics["latency"]:
        perf_text += "\n### Latency (CRITICAL PERFORMANCE METRIC)\n"
        perf_text += "The latency metrics below show the processing delay in clock cycles, which directly impacts overall system performance and throughput.\n\n"
        
        # Check if we have the new dictionary format or old single value
        first_latency = next(iter(metrics["latency"].values()))
        if isinstance(first_latency, dict):
            # Create table header for the dictionary format - enhanced for clarity
            perf_text += "| Implementation | Min Latency (cycles) | Max Latency (cycles) | Average Latency (cycles) | Throughput (samples/cycle) | Interval Min |\n"
            perf_text += "|----------------|---------------------|---------------------|--------------------------|----------------------------|-------------|\n"
            
            for impl, latency in metrics["latency"].items():
                min_val = latency.get('min', '-')
                max_val = latency.get('max', '-')
                avg_val = latency.get('average', '-')
                interval_min = latency.get('interval_min', '-')
                
                # Handle throughput which might be a float
                throughput = latency.get('throughput', '-')
                if throughput != '-':
                    try:
                        if isinstance(throughput, str):
                            throughput = float(throughput)
                        throughput_str = f"{throughput:.6f}"
                    except (ValueError, TypeError):
                        throughput_str = throughput
                else:
                    throughput_str = "-"
                
                perf_text += f"| {impl} | {min_val} | {max_val} | {avg_val} | {throughput_str} | {interval_min} |\n"
        else:
            # Simple table for the old format
            perf_text += "| Implementation | Latency (cycles) | Estimated Time @ Target Freq |\n"
            perf_text += "|----------------|------------------|------------------------------|\n"
            
            for impl, latency in metrics["latency"].items():
                # Try to estimate actual time if we have timing info
                time_estimate = "-"
                if "timing" in metrics and impl in metrics["timing"]:
                    target_freq = metrics["timing"][impl].get("Target_MHz", None)
                    if target_freq:
                        # Extract numeric value from string if needed
                        if isinstance(target_freq, str) and "MHz" in target_freq:
                            try:
                                freq_val = float(target_freq.split()[0])
                                time_ns = (latency / freq_val) * 1000
                                time_estimate = f"{time_ns:.2f} ns"
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(target_freq, (int, float)) and target_freq > 0:
                            time_ns = (latency / target_freq) * 1000
                            time_estimate = f"{time_ns:.2f} ns"
                
                perf_text += f"| {impl} | {latency} | {time_estimate} |\n"
        
        perf_text += "\nNote: Latency is a key performance metric that determines how quickly results can be produced.\n"
        perf_text += "Lower latency values generally indicate better performance, while throughput indicates how many operations can be processed per cycle.\n"
    
    return perf_text

def create_documentation_prompt(workflow_data, metrics, component_dir, context, output_format, llm_insights=None):
    """Create a detailed prompt for the LLM to generate documentation"""
    # ...existing code...
    
    # Generate performance metrics text directly from metrics object
    if metrics and (not isinstance(metrics, dict) or "error" not in metrics):
        perf_text = format_metrics_as_text(metrics)
        
        # Add explicit instructions for timing and latency metrics
        perf_text += "\n## IMPORTANT: PERFORMANCE METRICS REQUIREMENTS\n"
        perf_text += "When creating documentation, you MUST include detailed sections on the following performance metrics:\n\n"
        
        perf_text += "1. **Timing Metrics**:\n"
        perf_text += "   - Include clock period (ns) and frequency (MHz)\n"
        perf_text += "   - Discuss timing constraints and their implications\n"
        perf_text += "   - Analyze how timing affects overall system performance\n"
        perf_text += "   - Compare target vs. achieved timing if available\n\n"
        
        perf_text += "2. **Latency Metrics**:\n"
        perf_text += "   - Report latency in clock cycles for various operations\n"
        perf_text += "   - Calculate time-based latency using frequency information\n"
        perf_text += "   - Discuss throughput and its relationship to latency\n"
        perf_text += "   - Analyze pipeline behavior and initiation intervals if applicable\n\n"
        
        perf_text += "3. **Performance Visualizations**:\n"
        perf_text += "   - Create tables showing timing and latency measurements\n"
        perf_text += "   - If multiple implementations exist, include comparative charts\n"
        perf_text += "   - Use Mermaid diagrams to illustrate performance characteristics\n"
    else:
        # Fallback if metrics are not available
        perf_text = "No performance metrics are available."
    
    # ...existing code...

# ...existing code...
