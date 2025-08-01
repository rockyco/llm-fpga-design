#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import subprocess
import argparse
import shutil
import re
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent_log.txt")]
)
logger = logging.getLogger("fpga_agents")

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
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Get a prompt template by name, loading from file if needed"""
        if prompt_name in self.prompts_cache:
            # Update usage count
            self.usage_history[prompt_name] = self.usage_history.get(prompt_name, 0) + 1
            return self.prompts_cache[prompt_name]
        
        # Try different extensions
        extensions = [".md", ".txt", ""]
        for ext in extensions:
            prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}{ext}")
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, 'r') as f:
                        prompt_content = f.read()
                    
                    # Cache the prompt
                    self.prompts_cache[prompt_name] = prompt_content
                    self.usage_history[prompt_name] = 1
                    logger.debug(f"Loaded prompt: {prompt_name} from {prompt_path}")
                    return prompt_content
                except Exception as e:
                    logger.error(f"Error loading prompt {prompt_path}: {e}")
        
        logger.warning(f"Prompt not found: {prompt_name}")
        return None
    
    def update_prompt(self, prompt_name: str, new_content: str, metadata: Dict = None) -> bool:
        """Update a prompt template with new content and optional metadata"""
        if not prompt_name:
            logger.error("Cannot update prompt: no prompt name provided")
            return False
            
        # Find prompt file path
        prompt_path = None
        extensions = [".md", ".txt", ""]
        for ext in extensions:
            test_path = os.path.join(self.prompts_dir, f"{prompt_name}{ext}")
            if os.path.exists(test_path):
                prompt_path = test_path
                break
        
        if not prompt_path:
            # Create new file with .md extension
            prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.md")
        
        try:
            # Create backup before updating
            if os.path.exists(prompt_path):
                backup_dir = os.path.join(self.prompts_dir, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"{prompt_name}_{timestamp}.md")
                shutil.copy2(prompt_path, backup_path)
                logger.info(f"Created prompt backup: {backup_path}")
            
            # Add metadata as comments if provided
            if metadata:
                metadata_str = "<!-- Prompt Metadata\n"
                for key, value in metadata.items():
                    metadata_str += f"{key}: {value}\n"
                metadata_str += "Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
                metadata_str += "-->\n\n"
                new_content = metadata_str + new_content
            
            # Write updated content
            with open(prompt_path, 'w') as f:
                f.write(new_content)
            
            # Update cache
            self.prompts_cache[prompt_name] = new_content
            logger.info(f"Updated prompt: {prompt_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_name}: {e}")
            return False
    
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

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"

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
    
    def run(self, context: Dict) -> Dict:
        """Run the agent's task - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run()")
    
    def update_status(self, status: AgentStatus, result=None, error=None):
        """Update the agent's status and results"""
        self.status = status
        if result is not None:
            self.last_result = result
        if error is not None:
            self.error_message = error
            logger.error(f"Agent {self.name}: {error}")
        
        status_msg = f"Agent {self.name} status: {self.status.value}"
        if self.status == AgentStatus.FAILED and error:
            status_msg += f" - {error}"
        logger.info(status_msg)
        
        # Update used prompts with execution feedback if available
        if self.prompt_manager and self.used_prompts:
            for prompt_name in self.used_prompts:
                feedback_result = {
                    "status": self.status.value,
                    "result": self.last_result,
                    "error": self.error_message
                }
                
                # Try to update prompt with execution feedback
                self.prompt_manager.enrich_prompt_with_feedback(
                    prompt_name, 
                    feedback_result
                )
        
        return {
            "agent": self.name,
            "status": self.status.value,
            "result": self.last_result,
            "error": self.error_message
        }
    
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
        
        try:
            # Build command line args from context
            cmd = [sys.executable, self.script_path]
            for key, value in context.get("args", {}).items():
                if isinstance(value, list):
                    cmd.append(f"--{key}")
                    cmd.extend([str(item) for item in value])
                elif value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the script with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            
            # Process output
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            # Clean up temporary prompt file if we created one
            if "temp_prompt_path" in locals():
                try:
                    os.unlink(temp_prompt_path)
                except:
                    pass
            
            self.update_status(AgentStatus.SUCCESS, result=output)
            return {"status": "success", "output": output}
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Command timed out after {self.timeout} seconds"
            self.update_status(AgentStatus.FAILED, error=error_msg)
            return {"status": "failed", "error": error_msg, "timeout": True}
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with return code {e.returncode}: {e.stderr}"
            self.update_status(AgentStatus.FAILED, error=error_msg)
            return {"status": "failed", "error": error_msg, "output": e.stdout}
        
        except Exception as e:
            error_msg = f"Error running script: {str(e)}"
            self.update_status(AgentStatus.FAILED, error=error_msg)
            return {"status": "failed", "error": error_msg}
        
        finally:
            # Clean up temporary prompt file if we created one
            if "temp_prompt_path" in locals():
                try:
                    os.unlink(temp_prompt_path)
                except:
                    pass

class MakefileAgent(BaseAgent):
    """Agent that runs make commands for HLS projects"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
    
    def run(self, context: Dict) -> Dict:
        """Run make commands based on context"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            # Get working directory and target from context
            work_dir = context.get("work_dir")
            target = context.get("target", "all")
            
            if not work_dir or not os.path.exists(work_dir):
                raise ValueError(f"Invalid working directory: {work_dir}")
            
            # Build and run make command
            cmd = ["make", target]
            env = os.environ.copy()
            
            # Add any environment variables from context
            if "env_vars" in context:
                for key, value in context["env_vars"].items():
                    env[key] = str(value)
            
            logger.info(f"Running make in {work_dir}: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                capture_output=True,
                text=True
            )
            
            # Check for errors in output
            if result.returncode != 0:
                self.update_status(AgentStatus.FAILED, 
                                  error=f"Make failed with return code {result.returncode}")
                return {
                    "status": "failed", 
                    "error": result.stderr,
                    "output": result.stdout
                }
            
            # Process output
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            # Look for common error patterns in stdout/stderr
            error_patterns = ["Error:", "error:", "failed", "Failed"]
            found_errors = []
            
            for pattern in error_patterns:
                if pattern in result.stdout or pattern in result.stderr:
                    # Extract lines containing errors
                    for line in result.stdout.splitlines() + result.stderr.splitlines():
                        if pattern in line:
                            found_errors.append(line.strip())
            
            if found_errors:
                error_msg = "\n".join(found_errors[:10])  # Limit to first 10 errors
                if len(found_errors) > 10:
                    error_msg += f"\n... and {len(found_errors) - 10} more errors"
                
                self.update_status(AgentStatus.FAILED, error=error_msg)
                return {
                    "status": "failed", 
                    "error": error_msg,
                    "errors_found": found_errors,
                    "output": output
                }
            
            self.update_status(AgentStatus.SUCCESS, result=output)
            return {"status": "success", "output": output}
            
        except Exception as e:
            error_msg = f"Error running make: {str(e)}"
            self.update_status(AgentStatus.FAILED, error=error_msg)
            return {"status": "failed", "error": error_msg}

class MakefileGeneratorAgent(BaseAgent):
    """Agent that generates Makefiles for HLS projects"""
    
    def __init__(self, name: str, description: str, template_makefile: str = None):
        super().__init__(name, description)
        self.template_makefile = template_makefile
    
    def run(self, context: Dict) -> Dict:
        """Generate a Makefile based on context and template"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            # Get component directory and name from context
            component_dir = context.get("work_dir")
            component_name = context.get("component")
            template_path = context.get("template_makefile", self.template_makefile)
            
            if not component_dir or not os.path.exists(component_dir):
                raise ValueError(f"Invalid component directory: {component_dir}")
            
            if not component_name:
                # Try to derive component name from directory
                component_name = os.path.basename(component_dir)
                
            logger.info(f"Generating Makefile for {component_name} in {component_dir}")
            
            # Create Makefile path
            makefile_path = os.path.join(component_dir, "Makefile")
            
            # If we have a template, copy and modify it
            if template_path and os.path.exists(template_path):
                with open(template_path, 'r') as template_file:
                    makefile_content = template_file.read()
                
                # Replace template values
                makefile_content = makefile_content.replace("DESIGN_NAME = peakPicker", f"DESIGN_NAME = {component_name}")
                
                # Write the Makefile
                with open(makefile_path, 'w') as makefile_file:
                    makefile_file.write(makefile_content)
                
                logger.info(f"Generated Makefile from template at {makefile_path}")
            else:
                # Create a basic Makefile from scratch
                with open(makefile_path, 'w') as makefile_file:
                    makefile_file.write(f"# Makefile for HLS Project\n\n")
                    makefile_file.write(f"# Set the design name\n")
                    makefile_file.write(f"DESIGN_NAME = {component_name}\n\n")
                    makefile_file.write("# Configuration variables\n")
                    makefile_file.write("CSIM = 1\n")
                    makefile_file.write("CSYNTH = 1\n")
                    makefile_file.write("COSIM = 1\n")
                    makefile_file.write("EXPORT_IP = 1\n")
                    makefile_file.write("IMPL = 1\n\n")
                    makefile_file.write("# Hardware configuration\n")
                    makefile_file.write("CLOCK_FREQ = 256\n")
                    makefile_file.write("FPGA_PART = xc7k410t-ffg900-2\n")
                    makefile_file.write("CLOCK_UNCERTAINTY = 12.5\n\n")
                    makefile_file.write("# Vitis HLS installation path - modify this to match your installation\n")
                    makefile_file.write("VITIS_HLS_PATH ?= /opt/Xilinx/Vitis_HLS/2023.2\n")
                    makefile_file.write("# HLS compiler and flags\n")
                    makefile_file.write("HLS = $(VITIS_HLS_PATH)/bin/vitis_hls\n\n")
                    makefile_file.write("HLS_PROJECT = proj_$(DESIGN_NAME)\n")
                    makefile_file.write("HLS_SOLUTION = solution1\n\n")
                    makefile_file.write("# Source files\n")
                    makefile_file.write("SRC_FILES = $(DESIGN_NAME).cpp\n")
                    makefile_file.write("TB_FILES = $(DESIGN_NAME)_tb.cpp\n")
                    makefile_file.write("TEST_DATA_DIR = ../../data\n")
                    makefile_file.write("TEST_DATA_FILES := $(wildcard $(TEST_DATA_DIR)/*.txt)\n\n")

                    # Add basic targets
                    makefile_file.write(".PHONY: all clean csim csynth cosim export_ip impl\n\n")
                    makefile_file.write("all: csim csynth\n\n")
                    
                    # Add csim target
                    makefile_file.write("# HLS C Simulation\n")
                    makefile_file.write("csim:\n")
                    makefile_file.write("ifeq ($(CSIM), 1)\n")
                    makefile_file.write("\t@echo \"Running HLS C Simulation...\"\n")
                    makefile_file.write("\t@echo \"open_project $(HLS_PROJECT)\" > csim.tcl\n")
                    makefile_file.write("\t@echo \"set_top $(DESIGN_NAME)\" >> csim.tcl\n")
                    makefile_file.write("\t@echo \"add_files $(SRC_FILES)\" >> csim.tcl\n")
                    makefile_file.write("\t@echo \"add_files -tb $(TB_FILES)\" >> csim.tcl\n")
                    makefile_file.write("\t@for file in $(TEST_DATA_FILES); do \\\n")
                    makefile_file.write("\t\techo \"add_files -tb $$file\" >> csim.tcl; \\\n")
                    makefile_file.write("\tdone\n")
                    makefile_file.write("\t@echo \"open_solution $(HLS_SOLUTION)\" >> csim.tcl\n")
                    makefile_file.write("\t@echo \"set_part {$(FPGA_PART)}\" >> csim.tcl\n")
                    makefile_file.write("\t@echo \"csim_design\" >> csim.tcl\n")
                    makefile_file.write("\t@echo \"exit\" >> csim.tcl\n")
                    makefile_file.write("\t$(HLS) -f csim.tcl\n")
                    makefile_file.write("endif\n\n")
                    
                    # Add more targets for csynth, cosim, export_ip...
                    
                    # Add impl target
                    makefile_file.write("# HLS Implementation\n")
                    makefile_file.write("impl:\n")
                    makefile_file.write("ifeq ($(IMPL), 1)\n")
                    makefile_file.write("\t@echo \"Running HLS Implementation...\"\n")
                    makefile_file.write("\t@echo \"open_project $(HLS_PROJECT)\" > impl.tcl\n")
                    makefile_file.write("\t@echo \"set_top $(DESIGN_NAME)\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"add_files $(SRC_FILES)\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"open_solution $(HLS_SOLUTION)\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"set_part {$(FPGA_PART)}\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"create_clock -period $(CLOCK_FREQ)MHz -name default\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"config_bind -effort high\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"config_schedule -effort high\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"config_compile -pipeline_loops 1\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"export_design -flow impl -rtl verilog\" >> impl.tcl\n")
                    makefile_file.write("\t@echo \"exit\" >> impl.tcl\n")
                    makefile_file.write("\t$(HLS) -f impl.tcl\n")
                    makefile_file.write("endif\n\n")
                    
                    # Add clean target
                    makefile_file.write("# Clean up\n")
                    makefile_file.write("clean:\n")
                    makefile_file.write("\t@echo \"Cleaning up...\"\n")
                    makefile_file.write("\trm -rf $(HLS_PROJECT) *.dat *.log *.tcl\n")

                logger.info(f"Generated basic Makefile at {makefile_path}")
            
            # Return success
            output = {
                "makefile_path": makefile_path,
                "component_name": component_name
            }
            
            self.update_status(AgentStatus.SUCCESS, result=output)
            return {"status": "success", "output": output}
            
        except Exception as e:
            error_msg = f"Error generating Makefile: {str(e)}"
            self.update_status(AgentStatus.FAILED, error=error_msg)
            return {"status": "failed", "error": error_msg}

class DocumentationAgent(BaseAgent):
    """Agent that generates documentation and research papers from FPGA implementation results"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.workflow_summary = {}
        self.error_collection = []
        self.debug_methods = []
        self.performance_metrics = {}
        self.llm_insights = {}  # Store insights extracted from LLM responses
    
    def analyze_reports(self, report_dir: str):
        """Extract performance metrics from HLS and implementation reports"""
        try:
            import pandas as pd
            import re
            from pathlib import Path
            import glob
            import os
            
            # Find implementation reports
            impl_reports = glob.glob(os.path.join(report_dir, '**/impl/report/verilog/export_impl.rpt'), recursive=True)
            latency_reports = glob.glob(os.path.join(report_dir, '**/sim/report/verilog/lat.rpt'), recursive=True)
            
            all_resources = {}
            all_timing = {}
            all_latency = {}
            
            # Parse implementation reports
            for report_file in impl_reports:
                impl_name = Path(report_file).parents[3].name
                resource_data, timing_data = self._parse_impl_report(report_file)
                all_resources[impl_name] = resource_data
                all_timing[impl_name] = timing_data
                
            # Parse latency reports
            for report_file in latency_reports:
                impl_name = Path(report_file).parents[3].name
                latency = self._parse_latency_report(report_file)
                if latency is not None:
                    all_latency[impl_name] = latency
            
            # Create structured metrics dictionary
            metrics = {
                "resources": all_resources,
                "timing": all_timing,
                "latency": all_latency
            }
            
            # Generate a markdown report file for analysis
            report_file_path = os.path.join(report_dir, "performance_metrics.md")
            self._generate_performance_report(metrics, report_file_path)
            
            self.performance_metrics = metrics
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error analyzing reports: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _parse_impl_report(self, report_file):
        """Parse the Vivado Place & Route report file."""
        resource_summary = {}
        timing_summary = {}

        try:
            with open(report_file, 'r') as f:
                content = f.read()

            # Extract Resource Summary
            resource_pairs = [
                ('LUT', r'LUT:\s*(\d+)'),
                ('FF', r'FF:\s*(\d+)'),
                ('DSP', r'DSP:\s*(\d+)'),
                ('BRAM', r'BRAM:\s*(\d+)'),
                ('URAM', r'URAM:\s*(\d+)'),
                ('SRL', r'SRL:\s*(\d+)')
            ]
            for name, pattern in resource_pairs:
                match = re.search(pattern, content)
                if match:
                    resource_summary[name] = int(match.group(1))

            # Extract Timing Summary
            timing_pairs = [
                ('Target', r'\| Target\s*\|\s*([\d.]+)\s*\|'),
                ('Post-Synthesis', r'\| Post-Synthesis\s*\|\s*([\d.]+)\s*\|'),
                ('Post-Route', r'\| Post-Route\s*\|\s*([\d.]+)\s*\|')
            ]
            for name, pattern in timing_pairs:
                match = re.search(pattern, content)
                if match:
                    timing_ns = float(match.group(1))
                    timing_summary[name] = timing_ns
                    # Also add MHz value
                    timing_summary[f"{name}_MHz"] = f"{1000.0 / timing_ns:.2f} MHz" if timing_ns > 0 else "0.00 MHz"
                    
        except Exception as e:
            logger.warning(f"Error parsing implementation report {report_file}: {e}")
            
        return resource_summary, timing_summary
    
    def _parse_latency_report(self, report_file):
        """Parse latency report file and extract total execution time."""
        try:
            with open(report_file, 'r') as f:
                content = f.read()
                
                # Try to extract more comprehensive latency information
                latency_dict = {}
                
                # Extract max latency
                max_match = re.search(r'\$MAX_LATENCY = "(\d+)"', content)
                if max_match:
                    latency_dict["max"] = int(max_match.group(1))
                
                # Extract min latency
                min_match = re.search(r'\$MIN_LATENCY = "(\d+)"', content)
                if min_match:
                    latency_dict["min"] = int(min_match.group(1))
                
                # Extract average latency
                avg_match = re.search(r'\$AVERAGE_LATENCY = "(\d+)"', content)
                if avg_match:
                    latency_dict["average"] = int(avg_match.group(1))
                
                # Extract interval information
                interval_match = re.search(r'\$INTERVAL_MIN = "(\d+)"', content)
                if interval_match:
                    latency_dict["interval_min"] = int(interval_match.group(1))
                
                interval_max_match = re.search(r'\$INTERVAL_MAX = "(\d+)"', content)
                if interval_max_match:
                    latency_dict["interval_max"] = int(interval_max_match.group(1))
                
                # Try to extract throughput
                if "interval_min" in latency_dict and latency_dict["interval_min"] > 0:
                    latency_dict["throughput"] = 1.0 / latency_dict["interval_min"]
                
                # For backward compatibility, return the old format if no dict entries found
                if not latency_dict and max_match:
                    return int(max_match.group(1))
                elif latency_dict:
                    return latency_dict
                
                return None
                
        except Exception as e:
            logger.warning(f"Error parsing latency report {report_file}: {e}")
        return None
    
    def _generate_performance_report(self, metrics, output_file):
        """Generate a detailed performance report in markdown format"""
        try:
            with open(output_file, 'w') as f:
                f.write("# Performance Metrics Report\n\n")
                
                # Write resource utilization
                if "resources" in metrics and metrics["resources"]:
                    f.write("## Resource Utilization\n\n")
                    
                    # Create a markdown table header
                    f.write("| Implementation | LUT | FF | DSP | BRAM | URAM | SRL |\n")
                    f.write("|---------------|-----|----|----|------|---------|-----|\n")
                    
                    # Add rows for each implementation
                    for impl, resources in metrics["resources"].items():
                        row = f"| {impl} "
                        for res_type in ["LUT", "FF", "DSP", "BRAM", "URAM", "SRL"]:
                            row += f"| {resources.get(res_type, '-')} "
                        row += "|\n"
                        f.write(row)
                    
                    f.write("\n")
                
                # Write timing information
                if "timing" in metrics and metrics["timing"]:
                    f.write("## Timing\n\n")
                    
                    # Create a markdown table header
                    f.write("| Implementation | Target (ns) | Target (MHz) | Post-Synthesis (ns) | Post-Synthesis (MHz) | Post-Route (ns) | Post-Route (MHz) |\n")
                    f.write("|---------------|------------|-------------|-------------------|---------------------|----------------|----------------|\n")
                    
                    # Add rows for each implementation
                    for impl, timing in metrics["timing"].items():
                        row = f"| {impl} "
                        
                        # Handle each timing metric safely
                        for metric in ["Target", "Post-Synthesis", "Post-Route"]:
                            # Get ns value, convert to float if needed
                            ns_val = timing.get(metric, '-')
                            if ns_val != '-':
                                try:
                                    if isinstance(ns_val, str):
                                        ns_val = float(ns_val)
                                    row += f"| {ns_val:.2f} "
                                except (ValueError, TypeError):
                                    row += f"| {ns_val} "
                            else:
                                row += "| - "
                            
                            # Get MHz value, convert to float if needed
                            mhz_key = f"{metric}_MHz"
                            mhz_val = timing.get(mhz_key, '-')
                            if mhz_val != '-':
                                try:
                                    if isinstance(mhz_val, str):
                                        # Extract numeric part if it's a string like "100.00 MHz"
                                        mhz_val = float(mhz_val.split()[0])
                                    row += f"| {mhz_val:.2f} "
                                except (ValueError, TypeError, IndexError):
                                    row += f"| {mhz_val} "
                            else:
                                row += "| - "
                        
                        row += "|\n"
                        f.write(row)
                    
                    f.write("\n")
                
                # Write latency information
                if "latency" in metrics and metrics["latency"]:
                    f.write("## Latency\n\n")
                    
                    # Check if we have the new dictionary format or old single value
                    first_latency = next(iter(metrics["latency"].values()))
                    if isinstance(first_latency, dict):
                        # Create table header for the dictionary format
                        f.write("| Implementation | Min (cycles) | Max (cycles) | Average (cycles) | Throughput (samples/cycle) |\n")
                        f.write("|---------------|-------------|-------------|-----------------|-----------------------------|\n")
                        
                        for impl, latency in metrics["latency"].items():
                            row = f"| {impl} "
                            row += f"| {latency.get('min', '-')} "
                            row += f"| {latency.get('max', '-')} "
                            row += f"| {latency.get('average', '-')} "
                            
                            # Handle throughput which might be a float
                            throughput = latency.get('throughput', '-')
                            if throughput != '-':
                                try:
                                    if isinstance(throughput, str):
                                        throughput = float(throughput)
                                    row += f"| {throughput:.6f} "
                                except (ValueError, TypeError):
                                    row += f"| {throughput} "
                            else:
                                row += "| - "
                            
                            row += "|\n"
                            f.write(row)
                    else:
                        # Simple table for the old format
                        f.write("| Implementation | Latency (cycles) |\n")
                        f.write("|---------------|----------------|\n")
                        
                        for impl, latency in metrics["latency"].items():
                            f.write(f"| {impl} | {latency} |\n")
                    
                    f.write("\n")
                                
                logger.info(f"Generated performance report at {output_file}")
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}") 

    def collect_workflow_data(self, history, context):
        """Collect and process data from workflow execution history"""
        self.workflow_summary = {
            "total_steps": len(history),
            "successful_steps": sum(1 for entry in history if entry["result"]["status"] == "success"),
            "error_steps": sum(1 for entry in history if entry["result"]["status"] == "failed"),
            "component_name": context.get("component", "unknown"),
            "generation_model": context.get("args", {}).get("model", "unknown") if "args" in context else "unknown"
        }
        
        # Collect error information
        for entry in history:
            if entry["result"]["status"] == "failed" and "error" in entry["result"]:
                self.error_collection.append({
                    "step": entry["step"],
                    "agent": entry["agent"],
                    "error": entry["result"]["error"]
                })
            
            # Collect debug methods when debug agent was successful
            if entry["agent"] == "debug_assistant" and entry["result"]["status"] == "success":
                if "output" in entry["result"] and "stdout" in entry["result"]["output"]:
                    # Try to extract debug methodologies from output
                    debug_output = entry["result"]["output"]["stdout"]
                    # Simple extraction - could be improved with more sophisticated parsing
                    self.debug_methods.append({
                        "step": entry["step"],
                        "debug_output": debug_output
                    })
        
        return {
            "workflow_summary": self.workflow_summary,
            "errors": self.error_collection,
            "debug_methods": self.debug_methods
        }
    
    def analyze_llm_responses(self, component_dir: str):
        """Analyze LLM responses stored in markdown files"""
        try:
            import glob
            import os
            import re
            from pathlib import Path
            
            llm_insights = {
                "code_generation": {},
                "debugging": {},
                "optimization": {}
            }
            
            # Look for LLM response files
            llm_response_file = os.path.join(component_dir, "llm_response.md")
            debug_report_dir = os.path.join(component_dir, "debug_reports")
            
            # Analyze code generation response
            if os.path.exists(llm_response_file):
                logger.info(f"Analyzing code generation LLM response: {llm_response_file}")
                with open(llm_response_file, 'r') as f:
                    content = f.read()
                
                # Extract algorithm insights
                algo_insights = self._extract_algorithm_insights(content)
                llm_insights["code_generation"]["algorithm_insights"] = algo_insights
                
                # Extract design decisions
                design_decisions = self._extract_design_decisions(content)
                llm_insights["code_generation"]["design_decisions"] = design_decisions
                
                # Extract implementation challenges
                impl_challenges = self._extract_implementation_challenges(content)
                llm_insights["code_generation"]["implementation_challenges"] = impl_challenges
            
            # Analyze debug reports
            if os.path.exists(debug_report_dir):
                logger.info(f"Analyzing debug reports in: {debug_report_dir}")
                debug_files = glob.glob(os.path.join(debug_report_dir, "*.md"))
                
                all_bugs = []
                all_fixes = []
                all_root_causes = []
                
                for debug_file in debug_files:
                    with open(debug_file, 'r') as f:
                        content = f.read()
                    
                    # Extract bug information
                    bugs = self._extract_bugs(content)
                    all_bugs.extend(bugs)
                    
                    # Extract fix information
                    fixes = self._extract_fixes(content)
                    all_fixes.extend(fixes)
                    
                    # Extract root cause analysis
                    root_causes = self._extract_root_causes(content)
                    all_root_causes.extend(root_causes)
                
                llm_insights["debugging"]["bugs"] = all_bugs
                llm_insights["debugging"]["fixes"] = all_fixes
                llm_insights["debugging"]["root_causes"] = all_root_causes
            
            # Analyze optimization reports if they exist
            opt_reports = glob.glob(os.path.join(component_dir, "optimization_*.md"))
            if opt_reports:
                for opt_file in opt_reports:
                    with open(opt_file, 'r') as f:
                        content = f.read()
                    
                    # Extract optimization strategies
                    strategies = self._extract_optimization_strategies(content)
                    llm_insights["optimization"]["strategies"] = strategies
                    
                    # Extract performance improvements
                    improvements = self._extract_performance_improvements(content)
                    llm_insights["optimization"]["improvements"] = improvements
            
            self.llm_insights = llm_insights
            return llm_insights
            
        except Exception as e:
            logger.error(f"Error analyzing LLM responses: {str(e)}")
            return {"error": str(e)}
    
    def _extract_algorithm_insights(self, content):
        """Extract algorithm insights from LLM response"""
        insights = []
        
        # Look for algorithm explanations
        # These patterns may need adjustment based on actual LLM output
        patterns = [
            r"Algorithm overview:(.*?)(?=##|\Z)",
            r"The algorithm works by:(.*?)(?=##|\Z)",
            r"Algorithm description:(.*?)(?=##|\Z)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                insight = match.strip()
                if insight and len(insight) > 20:  # Filter out very short matches
                    insights.append(insight)
        
        return insights
    
    def _extract_design_decisions(self, content):
        """Extract design decisions from LLM response"""
        decisions = []
        
        # Look for design decisions
        patterns = [
            r"Design decisions?:(.*?)(?=##|\Z)",
            r"I chose to:(.*?)(?=##|\Z)",
            r"Implementation decisions?:(.*?)(?=##|\Z)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract bullet points
                points = re.findall(r"(?:^|\n)\s*[*-]\s*(.*?)(?=\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if points:
                    decisions.extend([p.strip() for p in points if len(p.strip()) > 10])
                else:
                    # If no bullet points, take the whole paragraph
                    decision = match.strip()
                    if decision and len(decision) > 20:
                        decisions.append(decision)
        
        return decisions
    
    def _extract_implementation_challenges(self, content):
        """Extract implementation challenges from LLM response"""
        challenges = []
        
        # Look for challenges
        patterns = [
            r"Challenges?:(.*?)(?=##|\Z)",
            r"Implementation challenges?:(.*?)(?=##|\Z)",
            r"Difficulties encountered:(.*?)(?=##|\Z)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract bullet points
                points = re.findall(r"(?:^|\n)\s*[*-]\s*(.*?)(?=\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if points:
                    challenges.extend([p.strip() for p in points if len(p.strip()) > 10])
                else:
                    # If no bullet points, take the whole paragraph
                    challenge = match.strip()
                    if challenge and len(challenge) > 20:
                        challenges.append(challenge)
        
        return challenges
    
    def _extract_bugs(self, content):
        """Extract bugs from debug report"""
        bugs = []
        
        # Look for error analysis
        patterns = [
            r"Error Analysis.*?(?=\n##|\Z)(.*)",
            r"The following (?:error|bug)s? (?:was|were) found.*?(?=\n##|\Z)(.*)",
            r"Error Information.*?```(.*?)```"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract bullet points
                points = re.findall(r"(?:^|\n)\s*[*-]\s*(.*?)(?=\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if points:
                    bugs.extend([p.strip() for p in points if len(p.strip()) > 10])
                else:
                    # If no bullet points, look for "Error:" or similar in the text
                    error_lines = re.findall(r"(?:^|\n)(?:Error|Bug|Issue|Problem):\s*(.*?)(?=\n\n|\Z)", match, re.DOTALL | re.IGNORECASE)
                    if error_lines:
                        bugs.extend([e.strip() for e in error_lines if len(e.strip()) > 10])
                    else:
                        # Just take the whole paragraph if it's not too long
                        bug = match.strip()
                        if bug and 20 < len(bug) < 500:
                            bugs.append(bug)
        
        return bugs
    
    def _extract_fixes(self, content):
        """Extract fixes from debug report"""
        fixes = []
        
        # Look for solution/fix sections
        patterns = [
            r"Solution.*?(?=\n##|\Z)(.*)",
            r"Fix.*?(?=\n##|\Z)(.*)",
            r"The following changes were made.*?(?=\n##|\Z)(.*)",
            r"COMPLETE CORRECTED SOURCE CODE.*?(?=\n##|\Z)(.*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract bullet points
                points = re.findall(r"(?:^|\n)\s*[*-]\s*(.*?)(?=\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if points:
                    fixes.extend([p.strip() for p in points if len(p.strip()) > 10])
                else:
                    # Extract fix explanations
                    explanations = re.findall(r"(?:^|\n)(?:Changed|Updated|Fixed|Corrected|Added|Removed).*?(?=\n\n|\Z)", match, re.DOTALL)
                    if explanations:
                        fixes.extend([e.strip() for e in explanations if len(e.strip()) > 10])
        
        return fixes
    
    def _extract_root_causes(self, content):
        """Extract root cause analysis from debug report"""
        root_causes = []
        
        # Look for root cause sections
        patterns = [
            r"Root Cause.*?(?=\n##|\Z)(.*)",
            r"Analysis.*?(?=\n##|\Z)(.*)",
            r"The (?:issue|problem|bug|error) was caused by.*?(?=\n##|\Z)(.*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract causes
                causes = re.findall(r"(?:^|\n)(?:1\.|[*-])\s*(.*?)(?=\n\d\.|\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if causes:
                    root_causes.extend([c.strip() for c in causes if len(c.strip()) > 10])
                else:
                    # Just take the paragraph if it's not too long
                    cause = match.strip()
                    if cause and 20 < len(cause) < 500:
                        root_causes.append(cause)
        
        return root_causes
    
    def _extract_optimization_strategies(self, content):
        """Extract optimization strategies from optimization reports"""
        strategies = []
        
        # Look for optimization strategy sections
        patterns = [
            r"Optimization Strategies.*?(?=\n##|\Z)(.*)",
            r"Strategies.*?(?=\n##|\Z)(.*)",
            r"The following optimizations were applied.*?(?=\n##|\Z)(.*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract bullet points
                points = re.findall(r"(?:^|\n)\s*[*-]\s*(.*?)(?=\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if points:
                    strategies.extend([p.strip() for p in points if len(p.strip()) > 10])
                else:
                    # Extract strategy explanations
                    explanations = re.findall(r"(?:^|\n)(?:Applied|Used|Implemented).*?(?=\n\n|\Z)", match, re.DOTALL)
                    if explanations:
                        strategies.extend([e.strip() for e in explanations if len(e.strip()) > 10])
        
        return strategies
    
    def _extract_performance_improvements(self, content):
        """Extract performance improvements from optimization reports"""
        improvements = []
        
        # Look for performance improvement sections
        patterns = [
            r"Performance Improvements.*?(?=\n##|\Z)(.*)",
            r"Results.*?(?=\n##|\Z)(.*)",
            r"The following improvements were achieved.*?(?=\n##|\Z)(.*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Extract bullet points
                points = re.findall(r"(?:^|\n)\s*[*-]\s*(.*?)(?=\n[*-]|\n\n|\Z)", match, re.DOTALL)
                if points:
                    improvements.extend([p.strip() for p in points if len(p.strip()) > 10])
                else:
                    # Extract improvement statements with percentages or metrics
                    metrics = re.findall(r"(?:^|\n)(?:Reduced|Improved|Increased|Decreased).*?(?:\d+%|\d+x).*?(?=\n\n|\Z)", match, re.DOTALL)
                    if metrics:
                        improvements.extend([m.strip() for m in metrics if len(m.strip()) > 10])
        
        return improvements
    
    def run(self, context: Dict) -> Dict:
        """Run the documentation agent"""
        self.update_status(AgentStatus.RUNNING)
        
        try:
            # Extract parameters from context
            component_dir = context.get("component_dir")
            if not component_dir or not os.path.exists(component_dir):
                raise ValueError(f"Invalid component directory: {component_dir}")
                
            history = context.get("history", [])
            complete_context = context.get("complete_context", {})
            output_format = context.get("output_format", ["readme", "paper"])
            model = context.get("model", "gemini-2.5-pro-exp-03-25")
            
            # Step 1: Analyze reports if available
            logger.info("Analyzing performance reports...")
            metrics = self.analyze_reports(component_dir)
            
            # Store metrics for later use
            self.performance_metrics = metrics
            
            # Step 2: Analyze LLM responses for additional insights
            logger.info("Analyzing LLM responses for insights...")
            llm_insights = self.analyze_llm_responses(component_dir)
            
            # Step 3: Collect workflow execution data
            logger.info("Collecting workflow execution data...")
            workflow_data = self.collect_workflow_data(history, complete_context)
            
            # Step 4: Generate documentation using an LLM
            logger.info("Generating documentation...")
            documentation = self._generate_documentation(
                workflow_data, 
                metrics,  # Pass metrics directly
                component_dir,
                complete_context,
                output_format,
                model,
                llm_insights
            )
            
            if documentation and "error" not in documentation:
                self.update_status(AgentStatus.SUCCESS, result=documentation)
                return {"status": "success", "output": documentation}
            else:
                error_msg = documentation.get("error", "Unknown error generating documentation")
                self.update_status(AgentStatus.FAILED, error=error_msg)
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error in documentation agent: {str(e)}"
            self.update_status(AgentStatus.FAILED, error=error_msg)
            return {"status": "failed", "error": error_msg}
    
    def _generate_documentation(self, workflow_data, metrics, component_dir, context, output_format, model, llm_insights=None):
        """Generate documentation using an LLM"""
        try:
            import subprocess
            import tempfile
            import json
            
            # Create a prompt for the LLM with collected data
            prompt = self._create_documentation_prompt(
                workflow_data, 
                metrics,  # Pass metrics data directly
                component_dir, 
                context, 
                output_format, 
                llm_insights
            )
            
            # Write prompt to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
                temp_file.write(prompt)
                prompt_file = temp_file.name
            
            # Use a script to call the LLM (similar to debug_assistant.py or generate_hls_code.py)
            output_dir = os.path.join(component_dir, "documentation")
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Calling LLM with model {model} to generate documentation...")
            result = subprocess.run(
                [
                    sys.executable, 
                    os.path.join(os.path.dirname(__file__), "generate_documentation.py"),
                    "--prompt", prompt_file,
                    "--output_dir", output_dir,
                    "--model", model,
                    "--formats", *output_format
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode != 0:
                return {"error": f"LLM documentation generation failed: {result.stderr}"}
            
            # Get the output files (README.md and/or paper.md)
            output_files = {}
            for fmt in output_format:
                file_path = os.path.join(output_dir, f"{fmt}.md")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        output_files[fmt] = f.read()
            
            return {
                "files": output_files,
                "output_dir": output_dir,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "LLM documentation generation timed out"}
        except Exception as e:
            return {"error": f"Error generating documentation: {str(e)}"}
    
    def _create_documentation_prompt(self, workflow_data, metrics, component_dir, context, output_format, llm_insights=None):
        """Create a detailed prompt for the LLM to generate documentation"""
        # Read source files to include in documentation
        component = context.get("component", os.path.basename(component_dir))
        source_files = {}
        for ext in [".hpp", ".cpp", "_tb.cpp"]:
            file_path = os.path.join(component_dir, f"{component}{ext}")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    source_files[f"{component}{ext}"] = f.read()
        
        # Format error and debug information
        errors_text = ""
        for err in workflow_data.get("errors", []):
            errors_text += f"- Step '{err['step']}' (Agent '{err['agent']}'):\n  {err['error']}\n\n"
        
        debug_text = ""
        for debug in workflow_data.get("debug_methods", []):
            # Extract just the key insights from debug output (first 500 chars)
            debug_summary = debug.get("debug_output", "")[:500] + "..."
            debug_text += f"- Step '{debug['step']}']:\n  {debug_summary}\n\n"
        
        # Generate performance metrics text directly from metrics object
        # instead of trying to read from markdown file
        if metrics and not isinstance(metrics, dict) or "error" not in metrics:
            perf_text = self._format_metrics_as_text(metrics)
        else:
            # Fallback if metrics are not available
            perf_text = "No performance metrics are available."
        
        # Format LLM insights if available
        llm_insights_text = ""
        if llm_insights:
            llm_insights_text += "## LLM Design Insights\n\n"
            
            # Add code generation insights
            if llm_insights.get("code_generation"):
                code_gen = llm_insights["code_generation"]
                
                # Add algorithm insights
                if code_gen.get("algorithm_insights"):
                    llm_insights_text += "### Algorithm Insights\n"
                    for insight in code_gen["algorithm_insights"][:3]:  # Limit to top 3
                        llm_insights_text += f"- {insight}\n"
                    llm_insights_text += "\n"
                
                # Add design decisions
                if code_gen.get("design_decisions"):
                    llm_insights_text += "### Design Decisions\n"
                    for decision in code_gen["design_decisions"][:5]:  # Limit to top 5
                        llm_insights_text += f"- {decision}\n"
                    llm_insights_text += "\n"
                
                # Add implementation challenges
                if code_gen.get("implementation_challenges"):
                    llm_insights_text += "### Implementation Challenges\n"
                    for challenge in code_gen["implementation_challenges"][:3]:  # Limit to top 3
                        llm_insights_text += f"- {challenge}\n"
                    llm_insights_text += "\n"
            
            # Add debugging insights
            if llm_insights.get("debugging"):
                debugging = llm_insights["debugging"]
                
                # Add bugs
                if debugging.get("bugs"):
                    llm_insights_text += "### Bugs Identified\n"
                    for bug in debugging["bugs"][:5]:  # Limit to top 5
                        llm_insights_text += f"- {bug}\n"
                    llm_insights_text += "\n"
                
                # Add fixes
                if debugging.get("fixes"):
                    llm_insights_text += "### Applied Fixes\n"
                    for fix in debugging["fixes"][:5]:  # Limit to top 5
                        llm_insights_text += f"- {fix}\n"
                    llm_insights_text += "\n"
                
                # Add root causes
                if debugging.get("root_causes"):
                    llm_insights_text += "### Root Causes\n"
                    for cause in debugging["root_causes"][:3]:  # Limit to top 3
                        llm_insights_text += f"- {cause}\n"
                    llm_insights_text += "\n"
            
            # Add optimization insights
            if llm_insights.get("optimization"):
                optimization = llm_insights["optimization"]
                
                # Add optimization strategies
                if optimization.get("strategies"):
                    llm_insights_text += "### Optimization Strategies\n"
                    for strategy in optimization["strategies"][:5]:  # Limit to top 5
                        llm_insights_text += f"- {strategy}\n"
                    llm_insights_text += "\n"
                
                # Add performance improvements
                if optimization.get("improvements"):
                    llm_insights_text += "### Performance Improvements\n"
                    for improvement in optimization["improvements"][:3]:  # Limit to top 3
                        llm_insights_text += f"- {improvement}\n"
                    llm_insights_text += "\n"
        
        # Create the full prompt
        prompt = f"""# Documentation Generation Task

## Project Overview
You are tasked with generating documentation for an FPGA design project that was developed using an AI-assisted workflow.

Component name: {component}
Model used for generation: {workflow_data.get("workflow_summary", {}).get("generation_model", "unknown")}
Workflow steps: {workflow_data.get("workflow_summary", {}).get("total_steps", 0)}
Successful steps: {workflow_data.get("workflow_summary", {}).get("successful_steps", 0)}
Error steps: {workflow_data.get("workflow_summary", {}).get("error_steps", 0)}

## Requirements

Please generate the following documentation:
"""

        if "readme" in output_format:
            prompt += """
1. A comprehensive README.md file that includes:
   - Project overview and purpose
   - Design architecture and principles
   - Implementation details
   - Performance metrics and analysis
   - Setup and usage instructions
   - Challenges encountered and solutions applied
"""

        if "paper" in output_format:
            prompt += """
2. An academic research paper (in Markdown format) that includes:
   - Abstract
   - Introduction
   - Related Work
   - Methodology
   - Implementation
   - Experimental Results
   - Performance Analysis
   - Discussion of AI-assisted FPGA design methodologies
   - Conclusion
   - References
"""

        prompt += f"""
## Design Implementation Details

### Source Code
```cpp
{component}.hpp:
{source_files.get(f"{component}.hpp", "File not available")}
```

```cpp
{component}.cpp:
{source_files.get(f"{component}.cpp", "File not available")}
```

```cpp
{component}_tb.cpp:
{source_files.get(f"{component}_tb.cpp", "File not available")}
```

## Workflow Execution Information

### Errors Encountered
{errors_text if errors_text else "No significant errors were encountered during the workflow."}

### Debugging Methods
{debug_text if debug_text else "No debugging was required during the workflow."}

## Performance Metrics
{perf_text if perf_text else "No performance metrics are available."}

{llm_insights_text}

## Instructions
- Use an academic tone for the paper and a more accessible tone for the README.
- Focus on how this design was created using AI assistance.
- Highlight the strengths and limitations observed in this approach.
- Be specific and use concrete details from the provided information.
- When discussing performance, analyze the tradeoffs between area, timing, and latency.
- Incorporate all the LLM design insights when explaining the implementation and challenges.
"""

        # Add visualization instructions
        prompt += """
## Visualization Requirements

### Diagrams
Please include the following types of diagrams in your documentation using Mermaid syntax:

1. Architecture diagram showing the main components and their connections
2. Data flow diagram showing how data moves through the system
3. Algorithm flowchart showing the key processing steps
4. State diagram if the design includes state machines
5. Sequence diagram showing interaction between components

### Performance Visualization
Present performance data in clear, well-formatted tables:

1. Resource utilization table
2. Timing/frequency table  
3. Latency table with different test cases if applicable
4. Comparative analysis table if baseline data is available

Use markdown tables for all data presentations. If multiple implementations exist, include comparative tables.

### Mermaid Diagram Syntax
Use the following syntax for Mermaid diagrams:

```mermaid
diagram-type
  diagram content
```

Where diagram-type can be: flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, gantt, etc.
"""

        return prompt
        
    def _format_metrics_as_text(self, metrics):
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
        
        # Format timing information
        if "timing" in metrics and metrics["timing"]:
            perf_text += "\n### Timing\n"
            for impl, timing in metrics["timing"].items():
                perf_text += f"- **{impl}**: "
                for time_type, val in timing.items():
                    if "_MHz" in time_type:
                        continue  # Skip MHz entries as we format them alongside ns entries
                    
                    # Convert ns_val to float if it's a string
                    ns_val = val
                    if not isinstance(ns_val, (int, float)):
                        try:
                            ns_val = float(ns_val)
                        except (ValueError, TypeError):
                            # If conversion fails, use the original string value
                            perf_text += f"{time_type}: {ns_val}, "
                            continue
                    
                    # Get or calculate MHz value and ensure it's a float
                    mhz_val = timing.get(f"{time_type}_MHz", None)
                    if mhz_val is None:
                        # Calculate MHz from ns value
                        mhz_val = 1000.0/ns_val if ns_val > 0 else 0
                    elif isinstance(mhz_val, str):
                        # Try to extract numeric value from string (e.g., "100.00 MHz")
                        try:
                            mhz_val = float(mhz_val.split()[0])
                        except (ValueError, IndexError):
                            mhz_val = 0.0
                    
                    perf_text += f"{time_type}: {ns_val:.2f}ns ({mhz_val:.2f} MHz), "
                perf_text = perf_text.rstrip(", ") + "\n"
        
        # Format latency information
        if "latency" in metrics and metrics["latency"]:
            perf_text += "\n### Latency\n"
            for impl, latency in metrics["latency"].items():
                if isinstance(latency, dict):
                    perf_text += f"- **{impl}**: "
                    perf_text += f"Min: {latency.get('min', 'N/A')} cycles, "
                    perf_text += f"Max: {latency.get('max', 'N/A')} cycles, "
                    if 'average' in latency:
                        perf_text += f"Avg: {latency['average']} cycles, "
                    if 'throughput' in latency:
                        throughput = latency['throughput']
                        if isinstance(throughput, (int, float)):
                            perf_text += f"Throughput: {throughput:.6f} samples/cycle"
                        else:
                            perf_text += f"Throughput: {throughput} samples/cycle"
                    perf_text += "\n"
                else:
                    perf_text += f"- **{impl}**: {latency} cycles\n"
        
        return perf_text

class AgentOrchestrator:
    """Manages workflow and communication between agents"""
    
    def __init__(self):
        self.agents = {}
        self.workflow = {}
        self.context = {}
        self.history = []
        self.prompt_manager = PromptManager()  # Initialize prompt manager
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        agent.prompt_manager = self.prompt_manager  # Set prompt manager for agent
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def define_workflow(self, workflow: Dict):
        """Define a workflow with dependencies between agents"""
        self.workflow = workflow
        logger.info(f"Workflow defined with {len(workflow)} steps")
    
    def update_context(self, new_context: Dict):
        """Update the global context with new information"""
        self.context.update(new_context)
    
    def run_workflow(self, initial_context: Dict = None) -> bool:
        """Run the complete workflow from start to finish"""
        if initial_context:
            self.context = initial_context
        
        logger.info("Starting workflow execution")
        success = True
        
        # Add counters for debug cycles to prevent infinite loops
        debug_attempts = 0
        max_debug_attempts = 3  # Maximum number of consecutive debug attempts
        debug_cycle_in_progress = False
        last_error_step = None
        
        # Process each workflow step in order
        step_name = next(iter(self.workflow.keys()))  # Start with first step
        while step_name:
            step_config = self.workflow[step_name]
            logger.info(f"Executing workflow step: {step_name}")
            
            # Get agent and prepare context
            agent_name = step_config.get("agent")
            if agent_name not in self.agents:
                logger.error(f"Agent not found: {agent_name}")
                success = False
                break
            
            agent = self.agents[agent_name]
            
            # Prepare agent-specific context
            agent_context = {**self.context}  # Start with global context
            if "context" in step_config:
                agent_context.update(step_config["context"])
            
            # Run the agent
            logger.info(f"Running agent {agent_name} for step {step_name}")
            result = agent.run(agent_context)
            
            # Record history
            self.history.append({
                "step": step_name,
                "agent": agent_name,
                "result": result,
                "timestamp": time.time()
            })
            
            # Update global context with results if needed
            if "update_context" in step_config and result["status"] == "success":
                context_updates = {}
                for key, path in step_config["update_context"].items():
                    # Extract value from result using the specified path
                    value = result
                    for part in path.split('.'):
                        if part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    
                    if value is not None:
                        context_updates[key] = value
                
                self.update_context(context_updates)
            
            # Debug cycle handling
            next_step = None
            
            # Check if we're in a debug cycle
            if debug_cycle_in_progress:
                if step_name == "debug_errors" and agent.status == AgentStatus.SUCCESS:
                    # Debug succeeded, try build again
                    next_step = "build_csim"
                elif step_name == "build_csim":
                    if agent.status == AgentStatus.SUCCESS:
                        # Build succeeded after debugging, reset debug cycle
                        debug_cycle_in_progress = False
                        debug_attempts = 0
                        logger.info("Debug cycle completed successfully, continuing workflow")
                        # Continue with next step in normal workflow
                        if "next" in step_config:
                            next_step = step_config["next"] if isinstance(step_config["next"], str) else None
                    else:
                        # Build still failing after debug
                        debug_attempts += 1
                        logger.warning(f"Build still failing after debug attempt {debug_attempts}/{max_debug_attempts}")
                        
                        if debug_attempts >= max_debug_attempts:
                            # Too many failed debug attempts, go back to code generation
                            logger.warning("Maximum debug attempts reached, returning to code generation")
                            debug_cycle_in_progress = False
                            debug_attempts = 0
                            if "generate_code" in self.workflow:
                                next_step = "generate_code"
                            else:
                                logger.error("No generate_code step found in workflow")
                                success = False
                                break
                        else:
                            # Try debugging again
                            next_step = "debug_errors"
            
            # Regular workflow execution if not in debug cycle
            if not next_step:
                # Check if we should continue
                if agent.status == AgentStatus.FAILED:
                    # Check if this step has an error handler
                    if "on_error" in step_config:
                        error_step = step_config["on_error"]
                        
                        # Special case to handle "stop" directive
                        if error_step == "stop":
                            logger.info("Workflow stopped due to error and 'stop' directive")
                            success = False
                            break
                            
                        # Special case for build_csim errors to start debug cycle
                        if step_name == "build_csim" and error_step == "debug_errors":
                            debug_cycle_in_progress = True
                            last_error_step = step_name
                            next_step = "debug_errors"
                            logger.info("Starting debug cycle")
                        else:
                            logger.info(f"Encountered error, running error handling: {error_step}")
                            next_step = error_step
                    else:
                        # No error handling defined
                        success = False
                        break
                else:
                    # Step succeeded, get next step from workflow
                    if "next" in step_config:
                        if isinstance(step_config["next"], str):
                            # Simple string next step
                            next_step = step_config["next"]
                        elif isinstance(step_config["next"], dict):
                            # Conditional next steps
                            for condition, conditional_next in step_config["next"].items():
                                # Evaluate condition against context
                                if condition == "default":
                                    next_step = conditional_next
                                elif eval(condition, {"context": self.context}):
                                    next_step = conditional_next
                                    break
            
            # Prepare for next iteration or end
            if next_step and next_step in self.workflow:
                step_name = next_step
            else:
                if next_step and next_step not in self.workflow:
                    logger.error(f"Next step not found in workflow: {next_step}")
                    success = False
                break
        
        # After workflow completion, update prompts with feedback from the entire workflow
        self._update_prompts_from_workflow()
        
        logger.info(f"Workflow completed with status: {'success' if success else 'failed'}")
        return success
    
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
    
    def get_history(self) -> List[Dict]:
        """Get the execution history"""
        return self.history
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict]:
        """Get the current status of an agent"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            return {
                "name": agent.name,
                "status": agent.status.value,
                "last_result": agent.last_result,
                "error": agent.error_message
            }
        return None

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

def main():
    """Main function for running the agent framework"""
    parser = argparse.ArgumentParser(description="FPGA Design Agent Framework")
    parser.add_argument("--config", type=str, help="Path to workflow configuration JSON")
    parser.add_argument("--matlab_file", nargs='+', help="MATLAB reference file(s)")
    parser.add_argument("--prompt", type=str, help="Prompt template file")
    parser.add_argument("--output_dir", type=str, help="Output directory for generated code")
    parser.add_argument("--component", type=str, help="Component name")
    args = parser.parse_args()
    
    # Create orchestrator with standard agents
    orchestrator = create_standard_agents()
    
    # Load workflow from config file or use default
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            workflow = json.load(f)
        orchestrator.define_workflow(workflow)
    else:
        # Define default workflow if no config file provided
        component_name = args.component or "default_component"
        # Remove _tb suffix if present
        if component_name.endswith('_tb'):
            component_name = component_name[:-3]
        
        # Construct paths
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        matlab_file = args.matlab_file or []  # Ensure matlab_file is a list
        prompt_file = args.prompt or os.path.join(project_dir, "prompts", "hls_generation.md")
        output_dir = args.output_dir or os.path.join(project_dir, "implementations")
        
        default_workflow = {
            "generate_code": {
                "agent": "code_generator",
                "context": {
                    "args": {
                        "matlab_file": matlab_file,  # Pass the list directly
                        "prompt": prompt_file,
                        "output_dir": output_dir,
                        "model": "gemini-2.5-pro-exp-03-25"
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
                    "work_dir": os.path.join(output_dir, component_name),
                    "component": component_name,
                    "template_makefile": os.path.join(project_dir, "scripts", "Makefile")
                },
                "next": "build_csim",
                "on_error": "stop"
            },
            "build_csim": {
                "agent": "hls_builder",
                "context": {
                    "work_dir": os.path.join(output_dir, component_name),
                    "target": "csim"
                },
                "next": "build_csynth",
                "on_error": "debug_errors"
            },
            "debug_errors": {
                "agent": "debug_assistant",
                "context": {
                    "args": {
                        "error_log": os.path.join(output_dir, component_name, f"proj_{component_name}", "solution1", "csim", "report", f"{component_name}_csim.log"),
                        "source_file": [
                            os.path.join(output_dir, component_name, f"{component_name}.hpp"),
                            os.path.join(output_dir, component_name, f"{component_name}.cpp"),
                            os.path.join(output_dir, component_name, f"{component_name}_tb.cpp")
                        ],
                        "model": "gemini-2.5-pro-exp-03-25"
                    }
                },
                "next": "build_csim",
                "continue_on_success": True
            },
            "build_csynth": {
                "agent": "hls_builder",
                "context": {
                    "work_dir": os.path.join(output_dir, component_name),
                    "target": "csynth"
                },
                "next": "build_cosim",
                "on_error": "stop"
            },
            "build_cosim": {
                "agent": "hls_builder",
                "context": {
                    "work_dir": os.path.join(output_dir, component_name),
                    "target": "cosim"
                },
                "next": "export_ip",
                "on_error": "stop"
            },
            "export_ip": {
                "agent": "hls_builder",
                "context": {
                    "work_dir": os.path.join(output_dir, component_name),
                    "target": "export_ip"
                },
                "next": "build_impl",  # Add next step to implementation
                "on_error": "build_impl"  # Even proceed to implementation on error
            },
            # Add implementation step
            "build_impl": {
                "agent": "hls_builder",
                "context": {
                    "work_dir": os.path.join(output_dir, component_name),
                    "target": "impl"
                },
                "next": "generate_documentation",
                "on_error": "generate_documentation"
            },
            "generate_documentation": {
                "agent": "documentation_generator",
                "context": {
                    "component_dir": os.path.join(output_dir, component_name),
                    "component": component_name,
                    "output_format": ["readme", "paper"],
                    "model": "gemini-2.5-pro-exp-03-25"
                },
                "on_error": "stop"
            }
        }
        
        orchestrator.define_workflow(default_workflow)
    
    # Run the workflow
    initial_context = vars(args)
    success = orchestrator.run_workflow(initial_context)
    
    if success:
        logger.info("Agent workflow completed successfully")
    else:
        logger.error("Agent workflow failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
