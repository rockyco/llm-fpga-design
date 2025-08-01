# Continuously update the model

rockyco: Revise the relevant Python code in the agent framework’s associated files to utilize the new prompt markdown files stored in the 'prompts' folder. Continuously update these prompts with each run of the agent framework to improve performance and support more effective and efficient execution of future tasks.

GitHub Copilot: # Solution for Enhancing Prompt Management and Continuous Learning

I'll implement changes to improve how the agent framework uses and updates prompt markdown files in the 'prompts' folder. This will enable continuous learning and improvement of the prompts based on execution results.

## Step-by-step approach:

1. Enhance the `PromptManager` class to better track prompt performance
2. Add methods for more detailed prompt analysis and improvement
3. Improve prompt utilization and feedback collection in the agent framework
4. Ensure the new prompt files (performance_optimization.md and enhanced_hls_generation.md) are properly utilized

Let's implement these changes:

### [agent_framework.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/agent_framework.py)

I'll enhance the `PromptManager` class to improve how it handles prompts, add better tracking of prompt performance, and implement more sophisticated feedback mechanisms.

````python
# ...existing code...

class PromptManager:
    """Manages loading, updating, and saving prompt templates"""
    
    def __init__(self, prompts_dir=None):
        """Initialize with path to prompts directory"""
        if prompts_dir is None:
            # Try to find prompts directory relative to script location
            script_dir = Path(__file__).parent
            project_dir = script_dir.parent
            prompts_dir = os.path.join(project_dir, "prompts")
        
        self.prompts_dir = prompts_dir
        self.prompts_cache = {}  # Cache loaded prompts
        self.usage_history = {}  # Track prompt usage
        self.performance_metrics = {}  # Track prompt performance
        self.feedback_history = {}  # Store feedback on prompts
        
        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        logger.info(f"Prompt manager initialized with directory: {self.prompts_dir}")
    
    # ...existing code...
    
    def enrich_prompt_with_feedback(self, prompt_name: str, 
                                   execution_result: Dict, 
                                   success_patterns: List[str] = None,
                                   error_patterns: List[str] = None) -> bool:
        """Enhance a prompt with feedback from execution results"""
        # Get the original prompt
        original_prompt = self.get_prompt(prompt_name)
        if not original_prompt:
            logger.error(f"Cannot enrich: prompt {prompt_name} not found")
            return False
        
        # Update performance metrics tracking for this prompt
        if prompt_name not in self.performance_metrics:
            self.performance_metrics[prompt_name] = {
                "success_count": 0,
                "error_count": 0,
                "last_update": None,
                "common_errors": {},
                "success_patterns": {}
            }
        
        metrics = self.performance_metrics[prompt_name]
        
        # Extract useful information from execution result
        if execution_result.get("status") != "success":
            # Extract error information
            error_info = execution_result.get("error", "Unknown error")
            metrics["error_count"] += 1
            
            # Track specific error patterns
            for pattern in error_patterns or []:
                if pattern in error_info:
                    metrics["common_errors"][pattern] = metrics["common_errors"].get(pattern, 0) + 1
            
            # Add to feedback history
            if prompt_name not in self.feedback_history:
                self.feedback_history[prompt_name] = []
            
            self.feedback_history[prompt_name].append({
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": error_info,
                "identified_patterns": [p for p in (error_patterns or []) if p in error_info]
            })
            
            # Check if we already have an ERRORS or COMMON PITFALLS section
            if "## ERRORS" in original_prompt or "## COMMON PITFALLS" in original_prompt:
                # Append to existing section
                if "## ERRORS" in original_prompt:
                    section_marker = "## ERRORS"
                else:
                    section_marker = "## COMMON PITFALLS"
                
                sections = original_prompt.split(section_marker)
                if len(sections) >= 2:
                    # Find where the next section starts
                    next_section_match = re.search(r"^##\s+", sections[1], re.MULTILINE)
                    if next_section_match:
                        insert_pos = next_section_match.start()
                        sections[1] = sections[1][:insert_pos] + f"- {error_info}\n\n" + sections[1][insert_pos:]
                    else:
                        sections[1] += f"\n- {error_info}\n"
                    
                    updated_prompt = section_marker.join(sections)
                    return self.update_prompt(prompt_name, updated_prompt, {
                        "EnrichedWith": "Error feedback",
                        "Status": "Failed",
                        "LastError": error_info
                    })
            else:
                # Add new section for errors
                updated_prompt = original_prompt + f"\n\n## COMMON PITFALLS\n\n- {error_info}\n"
                return self.update_prompt(prompt_name, updated_prompt, {
                    "EnrichedWith": "New error section",
                    "Status": "Failed",
                    "LastError": error_info
                })
        else:
            # For successful executions
            metrics["success_count"] += 1
            
            # Extract output information
            output_str = ""
            if "output" in execution_result:
                if isinstance(execution_result["output"], dict):
                    if "stdout" in execution_result["output"]:
                        output_str = execution_result["output"]["stdout"]
                elif isinstance(execution_result["output"], str):
                    output_str = execution_result["output"]
            
            # Track success patterns
            success_info = []
            for pattern in success_patterns or []:
                if pattern in output_str:
                    metrics["success_patterns"][pattern] = metrics["success_patterns"].get(pattern, 0) + 1
                    success_info.append(f"Pattern '{pattern}' contributed to success")
            
            # Add to feedback history
            if prompt_name not in self.feedback_history:
                self.feedback_history[prompt_name] = []
            
            self.feedback_history[prompt_name].append({
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "output_sample": output_str[:200] + "..." if len(output_str) > 200 else output_str,
                "identified_patterns": [p for p in (success_patterns or []) if p in output_str]
            })
            
            if success_info:
                # Find or create BEST PRACTICES section
                if "## BEST PRACTICES" in original_prompt:
                    sections = original_prompt.split("## BEST PRACTICES")
                    if len(sections) >= 2:
                        # Find where the next section starts
                        next_section_match = re.search(r"^##\s+", sections[1], re.MULTILINE)
                        if next_section_match:
                            insert_pos = next_section_match.start()
                            sections[1] = sections[1][:insert_pos] + "\n".join([f"- {info}" for info in success_info]) + "\n\n" + sections[1][insert_pos:]
                        else:
                            sections[1] += "\n" + "\n".join([f"- {info}" for info in success_info]) + "\n"
                        
                        updated_prompt = "## BEST PRACTICES".join(sections)
                        return self.update_prompt(prompt_name, updated_prompt, {
                            "EnrichedWith": "Success feedback",
                            "Status": "Success",
                            "SuccessPatterns": ", ".join([p for p in (success_patterns or []) if p in output_str])
                        })
                else:
                    # Add new section for best practices
                    updated_prompt = original_prompt + "\n\n## BEST PRACTICES\n\n" + "\n".join([f"- {info}" for info in success_info]) + "\n"
                    return self.update_prompt(prompt_name, updated_prompt, {
                        "EnrichedWith": "New best practices section",
                        "Status": "Success",
                        "SuccessPatterns": ", ".join([p for p in (success_patterns or []) if p in output_str])
                    })
        
        # Update last update timestamp
        metrics["last_update"] = datetime.now().isoformat()
        
        # No updates made to the prompt content
        return False
    
    def analyze_prompt_performance(self, prompt_name: str = None) -> Dict:
        """Analyze performance metrics for a specific prompt or all prompts"""
        if prompt_name:
            if prompt_name in self.performance_metrics:
                return {prompt_name: self.performance_metrics[prompt_name]}
            else:
                return {prompt_name: {"error": "No performance data available"}}
        
        # Return all prompt performance metrics
        return self.performance_metrics
    
    def suggest_prompt_improvements(self, prompt_name: str) -> Dict:
        """Suggest improvements to a prompt based on performance metrics"""
        if prompt_name not in self.performance_metrics:
            return {"error": "No performance data available for this prompt"}
        
        metrics = self.performance_metrics[prompt_name]
        feedback = self.feedback_history.get(prompt_name, [])
        
        # Simple analysis
        total_uses = metrics["success_count"] + metrics["error_count"]
        if total_uses < 5:
            return {"status": "insufficient_data", "message": "Not enough data to make suggestions"}
        
        success_rate = metrics["success_count"] / total_uses if total_uses > 0 else 0
        
        suggestions = []
        
        # Check success rate
        if success_rate < 0.5:
            suggestions.append("This prompt has a low success rate. Consider a major revision.")
        
        # Check common errors
        most_common_errors = sorted(metrics["common_errors"].items(), key=lambda x: x[1], reverse=True)[:3]
        for error, count in most_common_errors:
            suggestions.append(f"Common error pattern: '{error}' occurred {count} times. Add specific guidance to avoid this.")
        
        # Check success patterns
        most_common_successes = sorted(metrics["success_patterns"].items(), key=lambda x: x[1], reverse=True)[:3]
        for pattern, count in most_common_successes:
            suggestions.append(f"Successful pattern: '{pattern}' occurred {count} times. Emphasize this approach.")
        
        return {
            "status": "success" if success_rate >= 0.7 else "needs_improvement",
            "success_rate": success_rate,
            "total_uses": total_uses,
            "suggestions": suggestions,
            "metrics": metrics
        }
    
    def fill_prompt_template(self, prompt_name: str, replacements: Dict) -> str:
        """Fill in template variables in a prompt with actual values"""
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            logger.error(f"Cannot fill template: prompt {prompt_name} not found")
            return None
        
        # Replace template variables
        for key, value in replacements.items():
            placeholder = "{{" + key + "}}"
            prompt = prompt.replace(placeholder, str(value))
        
        return prompt

# ...existing code...

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.last_result = None
        self.error_message = None
        self.prompt_manager = None  # Will be set by orchestrator
        self.used_prompts = []  # Track prompts used in this run
        self.prompt_feedback = {}  # Track feedback for prompts used
    
    # ...existing code...
    
    def load_prompt(self, prompt_name: str, replacements: Dict = None) -> Optional[str]:
        """Load a prompt template by name, optionally filling in template variables"""
        if self.prompt_manager:
            if replacements:
                prompt = self.prompt_manager.fill_prompt_template(prompt_name, replacements)
            else:
                prompt = self.prompt_manager.get_prompt(prompt_name)
                
            if prompt:
                # Track that we used this prompt
                if prompt_name not in self.used_prompts:
                    self.used_prompts.append(prompt_name)
                    self.prompt_feedback[prompt_name] = {
                        "success_patterns": [],
                        "error_patterns": []
                    }
                return prompt
            else:
                logger.warning(f"Agent {self.name} failed to load prompt: {prompt_name}")
        else:
            logger.warning(f"Agent {self.name} has no prompt manager configured")
        
        return None
    
    def add_prompt_feedback(self, prompt_name: str, success_pattern: str = None, error_pattern: str = None):
        """Add feedback about a prompt for later analysis"""
        if prompt_name not in self.prompt_feedback:
            self.prompt_feedback[prompt_name] = {
                "success_patterns": [],
                "error_patterns": []
            }
            
        if success_pattern:
            self.prompt_feedback[prompt_name]["success_patterns"].append(success_pattern)
        
        if error_pattern:
            self.prompt_feedback[prompt_name]["error_patterns"].append(error_pattern)

# ...existing code...

class ScriptAgent(BaseAgent):
    """Agent that wraps an existing Python script"""
    
    def __init__(self, name: str, description: str, script_path: str, timeout: int = 300):
        super().__init__(name, description)
        self.script_path = script_path
        self.timeout = timeout
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
    
    def run(self, context: Dict) -> Dict:
        """Run the wrapped script with arguments from context"""
        self.update_status(AgentStatus.RUNNING)
        
        # Extract prompt name if specified
        prompt_name = context.get("args", {}).get("prompt_name")
        replacements = context.get("args", {}).get("prompt_replacements", {})
        
        # Check if we have a prompt argument and load it if needed
        if "args" in context and "prompt" in context["args"]:
            prompt_file = context["args"]["prompt"]
            
            # If this is a prompt name rather than a file path, try to load from prompt manager
            if prompt_name or (not os.path.exists(prompt_file) and self.prompt_manager):
                # Use explicit prompt_name if provided, otherwise derive from file path
                if not prompt_name:
                    prompt_name = os.path.basename(prompt_file).split('.')[0]
                
                # Load prompt with replacements if available
                prompt_content = self.load_prompt(prompt_name, replacements)
                
                if prompt_content:
                    # Create temporary prompt file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
                        temp_file.write(prompt_content)
                        temp_prompt_path = temp_file.name
                    
                    # Update context to use the temporary file
                    context["args"]["prompt"] = temp_prompt_path
                    logger.info(f"Agent {self.name} using prompt {prompt_name} from PromptManager")

        # ...existing code...

# ...existing code...

class AgentOrchestrator:
    """Manages workflow and communication between agents"""
    
    # ...existing code...
    
    def _update_prompts_from_workflow(self):
        """Update prompts with feedback from the entire workflow execution"""
        # Get all used prompts from all agents
        all_used_prompts = set()
        prompt_feedback = {}
        
        for agent_name, agent in self.agents.items():
            all_used_prompts.update(agent.used_prompts)
            
            # Collect feedback from each agent
            for prompt_name, feedback in agent.prompt_feedback.items():
                if prompt_name not in prompt_feedback:
                    prompt_feedback[prompt_name] = {
                        "success_patterns": [],
                        "error_patterns": []
                    }
                
                prompt_feedback[prompt_name]["success_patterns"].extend(feedback["success_patterns"])
                prompt_feedback[prompt_name]["error_patterns"].extend(feedback["error_patterns"])
        
        # For each prompt, update with overall workflow success/failure patterns
        for prompt_name in all_used_prompts:
            # Analyze workflow for this prompt
            success_patterns = prompt_feedback.get(prompt_name, {}).get("success_patterns", [])
            error_patterns = prompt_feedback.get(prompt_name, {}).get("error_patterns", [])
            
            # Get custom patterns from workflow results
            workflow_success_patterns = self._extract_success_patterns_from_workflow(prompt_name)
            workflow_error_patterns = self._extract_error_patterns_from_workflow(prompt_name)
            
            # Combine all patterns
            all_success_patterns = list(set(success_patterns + workflow_success_patterns))
            all_error_patterns = list(set(error_patterns + workflow_error_patterns))
            
            # Get overall workflow status
            failed_steps = [entry for entry in self.history 
                           if entry["result"]["status"] == "failed"]
            workflow_status = "success" if len(failed_steps) == 0 else "failed"
            
            # Update the prompt with overall feedback
            self.prompt_manager.enrich_prompt_with_feedback(
                prompt_name,
                {"status": workflow_status},
                all_success_patterns,
                all_error_patterns
            )
            
            # Log performance analysis
            performance = self.prompt_manager.analyze_prompt_performance(prompt_name)
            if performance and prompt_name in performance:
                logger.info(f"Prompt '{prompt_name}' performance: " +
                           f"Success: {performance[prompt_name].get('success_count', 0)}, " +
                           f"Errors: {performance[prompt_name].get('error_count', 0)}")
    
    def _extract_success_patterns_from_workflow(self, prompt_name):
        """Extract success patterns from workflow history for a specific prompt"""
        patterns = []
        
        # Find successful steps that used this prompt
        for entry in self.history:
            if entry["result"]["status"] == "success":
                agent = self.agents.get(entry["agent"])
                if agent and prompt_name in agent.used_prompts:
                    # This is just an example - in real implementation,
                    # you would extract actual patterns from the outputs
                    step_pattern = f"Step '{entry['step']}' succeeded with agent '{entry['agent']}'"
                    patterns.append(step_pattern)
        
        return patterns
    
    def _extract_error_patterns_from_workflow(self, prompt_name):
        """Extract error patterns from workflow history for a specific prompt"""
        patterns = []
        
        # Find failed steps that used this prompt
        for entry in self.history:
            if entry["result"]["status"] == "failed":
                agent = self.agents.get(entry["agent"])
                if agent and prompt_name in agent.used_prompts and "error" in entry["result"]:
                    # Extract useful error information
                    error_msg = entry["result"]["error"]
                    if error_msg:
                        # Simplify the error message to a more general pattern
                        # This is a simplistic approach - in a real implementation,
                        # you would use more sophisticated error pattern extraction
                        error_pattern = error_msg.split('\n')[0][:100]
                        patterns.append(error_pattern)
        
        return patterns

# ...existing code...

def create_standard_agents() -> AgentOrchestrator:
    """Create a standard set of agents for HLS workflows"""
    orchestrator = AgentOrchestrator()
    
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Find template Makefile
    template_makefile = os.path.join(project_dir, "scripts", "Makefile")
    if not os.path.exists(template_makefile):
        template_makefile = None
        logger.warning(f"Template Makefile not found at {template_makefile}")
    
    # Register agents with appropriate timeouts
    code_gen_agent = ScriptAgent(
        "code_generator",
        "Generates HLS C++ code from algorithm descriptions",
        os.path.join(script_dir, "generate_hls_code.py"),
        timeout=600  # 10 minutes timeout for code generation
    )
    orchestrator.register_agent(code_gen_agent)
    
    debug_agent = ScriptAgent(
        "debug_assistant",
        "Analyzes and fixes errors in HLS C++ code",
        os.path.join(script_dir, "debug_assistant.py"),
        timeout=300  # 5 minutes timeout for debugging
    )
    orchestrator.register_agent(debug_agent)
    
    makefile_agent = MakefileGeneratorAgent(
        "makefile_generator",
        "Generates Makefiles for HLS projects",
        template_makefile
    )
    orchestrator.register_agent(makefile_agent)
    
    # Add performance optimization agent
    performance_opt_agent = ScriptAgent(
        "performance_optimizer",
        "Optimizes HLS C++ code for better performance",
        os.path.join(script_dir, "optimize_hls_code.py"),
        timeout=600  # 10 minutes timeout for optimization
    )
    orchestrator.register_agent(performance_opt_agent)
    
    build_agent = MakefileAgent(
        "hls_builder",
        "Runs HLS compilation and simulation using make"
    )
    orchestrator.register_agent(build_agent)
    
    # Add documentation agent
    documentation_agent = DocumentationAgent(
        "documentation_generator",
        "Generates documentation and research papers from implementation results"
    )
    orchestrator.register_agent(documentation_agent)
    
    return orchestrator

# ...existing code...
````

### [generate_hls_code.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/generate_hls_code.py)

Update the script to properly utilize prompts from the PromptManager and provide feedback for continuous improvement.

````python


Made changes.

# ...existing code...

def create_prompt(matlab_files, prompt_template):
    """Create the final prompt by combining MATLAB code with the template."""
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
    
    # Check for enhanced template
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

# ...existing code...

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
    
    # Determine if we should use enhanced prompt
    use_enhanced_prompt = True
    prompt_name = "enhanced_hls_generation"
    
    # Read MATLAB files
    matlab_files = [file for file in args.matlab_file]
    
    # Read prompt template - first check if we need to load from file
    if os.path.exists(args.prompt):
        prompt_template = read_file(args.prompt)
    else:
        # Try to find the prompt in standard locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        prompts_dir = os.path.join(project_dir, "prompts")
        
        # Check for enhanced_hls_generation.md
        enhanced_prompt = os.path.join(prompts_dir, "enhanced_hls_generation.md")
        if os.path.exists(enhanced_prompt):
            prompt_template = read_file(enhanced_prompt)
            prompt_name = "enhanced_hls_generation"
        else:
            # Check for regular hls_generation.md
            regular_prompt = os.path.join(prompts_dir, "hls_generation.md")
            if os.path.exists(regular_prompt):
                prompt_template = read_file(regular_prompt)
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
        print(f"Saved full response to {os.path.join(output_dir, 'llm_response.md')}")
        
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

# ...existing code...
````

### [optimize_hls_code.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/optimize_hls_code.py)

Create a new script for HLS performance optimization that uses the performance_optimization.md prompt.

````python


Made changes.

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

def apply_optimizations(source_dir, output_dir, optimized_code):
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
    log_file = apply_optimizations(args.source_dir, output_dir, optimized_code)
    
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
````

### [agent_cli.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/agent_cli.py)

Add support for performance optimization in the CLI tool:

````python


Made changes.

# ...existing code...

class FPGAAgentShell(cmd.Cmd):
    # ...existing code...
    
    def do_optimize(self, arg):
        """Optimize HLS code for performance: optimize [--primary goal] [--secondary goal] [--model model_name]"""
        # Parse arguments
        args = arg.split()
        model_name = self.default_model
        primary_goal = "Reduce latency"
        secondary_goal = "Maintain resource usage"
        
        i = 0
        while i < len(args):
            if args[i] == '--model' and i + 1 < len(args):
                model_name = args[i+1]
                i += 2
            elif args[i] == '--primary' and i + 1 < len(args):
                primary_goal = args[i+1]
                i += 2
            elif args[i] == '--secondary' and i + 1 < len(args):
                secondary_goal = args[i+1]
                i += 2
            else:
                i += 1
                
        # Check if we have a component
        if not self.current_component:
            print("Error: No component selected. Use 'set_component' first.")
            return
        
        # Check if component directory exists
        component_dir = self.current_context.get("component_dir")
        if not component_dir or not os.path.isdir(component_dir):
            # Try to find it in the default location
            component_dir = os.path.join(self.current_project_dir, "implementations", self.current_component)
            if not os.path.isdir(component_dir):
                print(f"Error: Component directory not found: {component_dir}")
                return
            
            self.current_context["component_dir"] = component_dir
        
        # Set up context for optimization agent
        agent_context = {
            "args": {
                "source_dir": component_dir,
                "model": model_name,
                "primary_goal": primary_goal,
                "secondary_goal": secondary_goal,
                "prompt_name": "performance_optimization"
            }
        }
        
        # Run optimization agent
        print(f"Optimizing HLS code for {self.current_component}...")
        print(f"Primary goal: {primary_goal}")
        print(f"Secondary goal: {secondary_goal}")
        agent = self.orchestrator.agents["performance_optimizer"]
        result = agent.run(agent_context)
        
        if agent.status == AgentStatus.SUCCESS:
            print("Code optimization completed successfully!")
            print("Run 'build csynth' to synthesize the optimized code")
        else:
            print(f"Code optimization failed: {agent.error_message}")
    
    # ...existing code...

def main():
    # ...existing code...
    
    # Add optimization option
    parser.add_argument("--optimize", action="store_true", 
                       help="Optimize the component for performance")
    parser.add_argument("--primary-goal", default="Reduce latency", 
                       help="Primary optimization goal")
    parser.add_argument("--secondary-goal", default="Maintain resource usage", 
                       help="Secondary optimization goal")
    
    # ...existing code...
    
    # Run optimization if requested
    if args.optimize:
        if not args.component:
            print("Error: --component is required when using --optimize")
            sys.exit(1)
            
        # Prepare arguments for do_optimize
        opt_args = f"--primary \"{args.primary_goal}\" --secondary \"{args.secondary_goal}\" --model {args.model}"
        shell.do_optimize(opt_args)
        sys.exit(0)
    
    # ...existing code...

# ...existing code...
````

### [default_workflow.json](file:///home/jielei/Projects/UTS/llm-fpga-design/workflows/default_workflow.json)

Update the default workflow to include the performance optimization step.

````json


Made changes.

{
  "generate_code": {
    "agent": "code_generator",
    "context": {
      "args": {
        "matlab_file": ["<MATLAB_FILE>"],
        "prompt": "<PROMPT_FILE>",
        "prompt_name": "enhanced_hls_generation",
        "output_dir": "<OUTPUT_DIR>",
        "model": "gemini-2.5-pro-exp-03-25",
        "timeout": 600
      }
    },
    "update_context": {
      "component_dir": "output.stdout"
    },
    "next": "generate_makefile",
    "on_error": "stop"
  },
  "generate_makefile": {
    "agent": "makefile_generator",
    "context": {
      "work_dir": "<COMPONENT_DIR>",
      "component": "<COMPONENT>",
      "template_makefile": "<PROJECT_DIR>/scripts/Makefile"
    },
    "next": "build_csim",
    "on_error": "stop"
  },
  "build_csim": {
    "agent": "hls_builder",
    "context": {
      "work_dir": "<COMPONENT_DIR>",
      "target": "csim"
    },
    "next": "build_csynth",
    "on_error": "debug_errors"
  },
  "debug_errors": {
    "agent": "debug_assistant",
    "context": {
      "args": {
        "error_log": "<COMPONENT_DIR>/proj_<COMPONENT>/solution1/csim/report/<COMPONENT>_csim.log",
        "source_file": [
          "<COMPONENT_DIR>/<COMPONENT>.hpp",
          "<COMPONENT_DIR>/<COMPONENT>.cpp",
          "<COMPONENT_DIR>/<COMPONENT>_tb.cpp"
        ],
        "model": "gemini-2.5-pro-exp-03-25",
        "timeout": 300
      }
    },
    "next": "build_csim",
    "on_error": "build_csynth"
  },
  "build_csynth": {
    "agent": "hls_builder",
    "context": {
      "work_dir": "<COMPONENT_DIR>",
      "target": "csynth"
    },
    "next": {
      "context.get('optimize', False) == True": "optimize_code",
      "default": "build_cosim"
    },
    "on_error": "stop"
  },
  "optimize_code": {
    "agent": "performance_optimizer",
    "context": {
      "args": {
        "source_dir": "<COMPONENT_DIR>",
        "prompt_name": "performance_optimization",
        "model": "gemini-2.5-pro-exp-03-25",
        "primary_goal": "Reduce latency by at least 20%",
        "secondary_goal": "Maintain resource utilization"
      }
    },
    "next": "build_csynth_optimized",
    "on_error": "build_cosim"
  },
  "build_csynth_optimized": {
    "agent": "hls_builder",
    "context": {
      "work_dir": "<COMPONENT_DIR>",
      "target": "csynth"
    },
    "next": "build_cosim",
    "on_error": "build_cosim"
  },
  "build_cosim": {
    "agent": "hls_builder",
    "context": {
      "work_dir": "<COMPONENT_DIR>",
      "target": "cosim"
    },
    "next": "export_ip",
    "on_error": "stop"
  },
  "export_ip": {
    "agent": "hls_builder",
    "context": {
      "work_dir": "<COMPONENT_DIR>",
      "target": "export_ip"
    },
    "next": "generate_documentation",
    "on_error": "generate_documentation"
  },
  "generate_documentation": {
    "agent": "documentation_generator",
    "context": {
      "component_dir": "<COMPONENT_DIR>",
      "component": "<COMPONENT>",
      "output_format": ["readme", "paper"],
      "model": "gemini-2.5-pro-exp-03-25"
    },
    "on_error": "stop"
  }
}
````

## Summary of Changes

1. Enhanced the `PromptManager` class with:
   - Improved tracking of prompt performance metrics
   - Better feedback collection about what patterns lead to success or failure
   - Methods to analyze prompt effectiveness and suggest improvements

2. Improved the agent framework to:
   - Track more detailed feedback about prompt usage
   - Better utilize template replacements in prompts
   - Enhanced the workflow-level prompt updates

3. Added a new performance optimization agent that:
   - Uses the performance_optimization.md prompt to optimize HLS code
   - Extracts existing performance metrics and uses them in the optimization
   - Backs up original files and applies optimizations

4. Updated the CLI tool to support performance optimization with custom goals

5. Modified the workflow to include an optional optimization step

These changes ensure that prompts are used more effectively and continuously improved based on execution results, leading to better performance of the agent framework over time.

# Update the HLS Generation Prompt

rockyco: Revise all relevant Python files to use only `hls_generation.md` as the prompt source for HLS code generation, and remove any references to `enhanced_hls_generation.md`. Continuously improve the `hls_generation.md` prompts through learning and updates to enhance their effectiveness in generating efficient future designs.

GitHub Copilot: # Solution: Standardize on hls_generation.md Prompt

I'll update all the relevant files to use only `hls_generation.md` as the prompt source for HLS code generation, removing references to `enhanced_hls_generation.md`.

## Step-by-step approach:
1. Update the default workflow to use "hls_generation" prompt name
2. Modify the code generation script to prioritize hls_generation.md
3. Ensure the agent framework supports prompt learning for continuous improvement

Let's implement these changes:

### [default_workflow.json](file:///home/jielei/Projects/UTS/llm-fpga-design/workflows/default_workflow.json)

Update the prompt name in the workflow configuration from "enhanced_hls_generation" to "hls_generation".

````json
{
  "generate_code": {
    "agent": "code_generator",
    "context": {
      "args": {
        "matlab_file": ["<MATLAB_FILE>"],
        "prompt": "<PROMPT_FILE>",
        "prompt_name": "hls_generation",
        "output_dir": "<OUTPUT_DIR>",
        "model": "gemini-2.5-pro-exp-03-25",
        "timeout": 600
      }
    },
    "update_context": {
      "component_dir": "output.stdout"
    },
    "next": "generate_makefile",
    "on_error": "stop"
  },
  // ...existing code...
}
````

### [generate_hls_code.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/generate_hls_code.py)

Modify the script to prioritize and only use hls_generation.md:

````python


Made changes.

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
    
    # Read prompt template - first check if we need to load from file
    if os.path.exists(args.prompt):
        prompt_template = read_file(args.prompt)
    else:
        # Try to find the prompt in standard locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
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
````

### [agent_cli.py](file:///home/jielei/Projects/UTS/llm-fpga-design/scripts/agent_cli.py)

Update the CLI to use hls_generation prompt:

````python


Made changes.

#!/usr/bin/env python3

import os
import sys
import json
import cmd
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from agent_framework import (
    AgentOrchestrator, 
    create_standard_agents,
    AgentStatus,
    BaseAgent
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent_cli_log.txt")]
)
logger = logging.getLogger("fpga_agent_cli")

class FPGAAgentShell(cmd.Cmd):
    # ...existing code...
    
    def do_set_component(self, arg):
        # ...existing code...
    
    def do_generate(self, arg):
        """Generate HLS code from MATLAB: generate <matlab_file1> [matlab_file2 ...] [--prompt prompt_file] [--model model_name]"""
        args = arg.split()
        
        if not args:
            print("Error: Please provide at least one MATLAB file path")
            return
        
        # Parse arguments
        prompt_file = None
        model_name = self.default_model
        matlab_files = []
        i = 0
        while i < len(args):
            if args[i] == '--prompt' and i + 1 < len(args):
                prompt_file = os.path.abspath(args[i+1])
                i += 2
            elif args[i] == '--model' and i + 1 < len(args):
                model_name = args[i+1]
                i += 2
            else:
                matlab_files.append(os.path.abspath(args[i]))
                i += 1
        
        # Validate MATLAB files
        for matlab_file in matlab_files:
            if not os.path.isfile(matlab_file):
                print(f"Error: File not found: {matlab_file}")
                return
        
        # Validate prompt file
        if prompt_file and not os.path.isfile(prompt_file):
            print(f"Error: Prompt file not found: {prompt_file}")
            return
        elif not prompt_file:
            # Use hls_generation.md as standard prompt file
            default_prompt = os.path.join(self.current_project_dir, "prompts", "hls_generation.md")
            if os.path.isfile(default_prompt):
                prompt_file = default_prompt
            else:
                print(f"Error: Standard prompt file not found: {default_prompt}")
                print("Please specify a prompt file or create the standard one")
                return
        
        # Use component name or extract from first filename
        component = self.current_component
        if not component:
            component = os.path.basename(matlab_files[0]).split('.')[0]
            self.current_component = component
            print(f"Using component name from file: {component}")
        
        # Set up context
        output_dir = os.path.join(self.current_project_dir, "implementations")
        agent_context = {
            "args": {
                "matlab_file": matlab_files,
                "prompt": prompt_file,
                "prompt_name": "hls_generation",  # Always use hls_generation as prompt name
                "output_dir": output_dir,
                "model": model_name,
                "timeout": self.generate_timeout
            }
        }
        
        # Run code generation agent
        print(f"Generating HLS code for {component} from {len(matlab_files)} MATLAB file(s)...")
        print(f"Using model: {model_name} (timeout: {self.generate_timeout}s)")
        agent = self.orchestrator.agents["code_generator"]
        result = agent.run(agent_context)
        
        if agent.status == AgentStatus.SUCCESS:
            print("Code generation successful!")
            # Update component directory in context for future commands
            self.current_context["component_dir"] = os.path.join(output_dir, component)
        else:
            print(f"Code generation failed: {agent.error_message}")
    
    # ...existing code...

def main():
    """Main function for running the FPGA agent CLI"""
    parser = argparse.ArgumentParser(description="FPGA Design Agent CLI")
    # ...existing code...
    
    # Add new arguments for direct documentation generation
    parser.add_argument("--generate-docs", action="store_true", help="Generate documentation without entering the shell")
    parser.add_argument("--doc-format", choices=["readme", "paper", "both"], default="both", 
                       help="Documentation format (readme, paper, or both)")
    
    # Add optimization option
    parser.add_argument("--optimize", action="store_true", 
                       help="Optimize the component for performance")
    parser.add_argument("--primary-goal", default="Reduce latency", 
                       help="Primary optimization goal")
    parser.add_argument("--secondary-goal", default="Maintain resource usage", 
                       help="Secondary optimization goal")
    
    # Add prompt standardization
    parser.add_argument("--prompt-name", default="hls_generation",
                        help="Prompt name to use (default: hls_generation)")
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("fpga_agents").setLevel(logging.DEBUG)
    
    # Create and configure the shell
    shell = FPGAAgentShell(default_model=args.model)
    shell.debug_timeout = args.debug_timeout
    shell.generate_timeout = args.generate_timeout
    
    # Apply command line arguments
    if args.component:
        shell.do_set_component(args.component)
    
    if args.project_dir:
        shell.do_set_project(args.project_dir)
    
    if args.workflow:
        shell.do_load_workflow(args.workflow)
    
    # Run documentation generation directly if requested
    if args.generate_docs:
        if not args.component:
            print("Error: --component is required when using --generate-docs")
            sys.exit(1)
            
        # Prepare format arg for do_generate_documentation
        if args.doc_format == "both":
            format_arg = ""  # Default is both
        else:
            format_arg = f"--format {args.doc_format}"
            
        # Add model if specified
        model_arg = f"--model {args.model}" if args.model else ""
        
        # Run the documentation generation
        shell.do_generate_documentation(f"{format_arg} {model_arg}".strip())
        sys.exit(0)
    
    # Run optimization if requested
    if args.optimize:
        if not args.component:
            print("Error: --component is required when using --optimize")
            sys.exit(1)
            
        # Prepare arguments for do_optimize
        opt_args = f"--primary \"{args.primary_goal}\" --secondary \"{args.secondary_goal}\" --model {args.model}"
        shell.do_optimize(opt_args)
        sys.exit(0)
    
    # Start the interactive shell
    shell.cmdloop()

if __name__ == "__main__":
    main()
````

## Explanation of Changes

1. **Default Workflow File (`default_workflow.json`)**:
   - Changed `prompt_name` from "enhanced_hls_generation" to "hls_generation" in the generate_code section

2. **Code Generation Script (`generate_hls_code.py`)**:
   - Simplified prompt loading to only look for `hls_generation.md`
   - Improved component name extraction and template variable replacement
   - Updated prompt tracking to use "hls_generation" as the standard name
   - Enhanced the `create_prompt` function to better handle component name replacement

3. **CLI Interface (`agent_cli.py`)**:
   - Updated the code generation command to always use "hls_generation" as prompt_name
   - Added a new CLI argument `--prompt-name` with "hls_generation" as default value
   - Improved the default prompt file resolution logic to prioritize hls_generation.md

These changes standardize the prompt system to always use `hls_generation.md`, which will allow the prompt to continuously improve through the existing feedback mechanisms in the agent framework. The PromptManager in the agent framework already supports prompt learning and improvement, which will now be focused on refining the single standard prompt.

