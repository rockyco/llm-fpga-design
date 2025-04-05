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
    """Interactive command shell for the FPGA design agent system"""
    
    intro = """
    ==========================================================
    FPGA Design Agent System - Interactive CLI
    ==========================================================
    Type 'help' or '?' to list commands.
    Type 'quit' to exit.
    """
    prompt = "fpga-agent> "
    
    def __init__(self, default_model="gemini-2.5-pro-exp-03-25"):
        super().__init__()
        self.orchestrator = create_standard_agents()
        self.current_context = {}
        self.current_component = None
        self.current_project_dir = str(Path(__file__).parent.parent)
        self.workflow_file = None
        self.default_model = default_model
        self.debug_timeout = 300  # 5 minutes timeout for debugging
        self.generate_timeout = 600  # 10 minutes timeout for generation
    
    def do_status(self, arg):
        """Show current status of all agents"""
        for name, agent in self.orchestrator.agents.items():
            status = self.orchestrator.get_agent_status(name)
            print(f"Agent: {name}")
            print(f"  Status: {status['status']}")
            if status['error']:
                print(f"  Error: {status['error']}")
            print()
    
    def do_set_component(self, arg):
        """Set the current component name: set_component <name>"""
        if not arg:
            print("Error: Please provide a component name")
            return
        
        self.current_component = arg.strip()
        print(f"Current component set to: {self.current_component}")
        
        # Update context
        self.current_context["component"] = self.current_component
    
    def do_set_project(self, arg):
        """Set the project directory: set_project <directory>"""
        if not arg:
            print("Error: Please provide a project directory")
            return
        
        path = os.path.abspath(arg.strip())
        if not os.path.isdir(path):
            print(f"Error: Directory not found: {path}")
            return
        
        self.current_project_dir = path
        print(f"Project directory set to: {self.current_project_dir}")
    
    def do_load_workflow(self, arg):
        """Load a workflow configuration: load_workflow <json_file>"""
        if not arg:
            print("Error: Please provide a JSON workflow file path")
            return
        
        file_path = os.path.abspath(arg.strip())
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                workflow = json.load(f)
            
            self.orchestrator.define_workflow(workflow)
            self.workflow_file = file_path
            print(f"Loaded workflow from: {file_path}")
        except Exception as e:
            print(f"Error loading workflow: {str(e)}")
    
    def do_set_model(self, arg):
        """Set the LLM model to use: set_model <model_name>"""
        if not arg:
            print(f"Current model: {self.default_model}")
            return
        
        self.default_model = arg.strip()
        print(f"Model set to: {self.default_model}")
    
    def do_set_timeout(self, arg):
        """Set timeout values: set_timeout <operation> <seconds>
        
        Operations: debug, generate"""
        args = arg.split()
        if len(args) != 2:
            print("Error: Please provide both operation and seconds")
            print(f"Current timeouts - Debug: {self.debug_timeout}s, Generate: {self.generate_timeout}s")
            return
        
        operation, seconds = args
        try:
            seconds = int(seconds)
            if seconds < 10:
                print("Warning: Timeout too short, setting to minimum 10 seconds")
                seconds = 10
                
            if operation.lower() == "debug":
                self.debug_timeout = seconds
                print(f"Debug timeout set to {seconds} seconds")
            elif operation.lower() == "generate":
                self.generate_timeout = seconds
                print(f"Generate timeout set to {seconds} seconds")
            else:
                print(f"Unknown operation: {operation}. Valid operations: debug, generate")
        except ValueError:
            print(f"Error: Invalid timeout value: {seconds}")
    
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
            # Use default prompt file
            default_prompt = os.path.join(self.current_project_dir, "prompts", "hls_generation.md")
            if os.path.isfile(default_prompt):
                prompt_file = default_prompt
            else:
                print(f"Error: Default prompt file not found: {default_prompt}")
                print("Please specify a prompt file or create the default one")
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
                "matlab_file": matlab_files,  # Pass all MATLAB files
                "prompt": prompt_file,
                "output_dir": output_dir,
                "model": model_name,
                "timeout": self.generate_timeout  # Pass timeout parameter
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
    
    def do_build(self, arg):
        """Build the current component: build <target>
        
        Targets: csim, csynth, cosim, export_ip, impl, or all"""
        if not arg:
            print("Error: Please specify a build target (csim, csynth, cosim, export_ip, impl, all)")
            return
        
        target = arg.strip()
        valid_targets = ["csim", "csynth", "cosim", "export_ip", "impl", "all"]
        if target not in valid_targets:
            print(f"Error: Invalid target '{target}'. Must be one of: {', '.join(valid_targets)}")
            return
        
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
                print("Generate the code first using the 'generate' command")
                return
            
            self.current_context["component_dir"] = component_dir
        
        # Set up context for build agent
        agent_context = {
            "work_dir": component_dir,
            "target": target
        }
        
        # Run build agent
        print(f"Building target '{target}' for component {self.current_component}...")
        agent = self.orchestrator.agents["hls_builder"]
        result = agent.run(agent_context)
        
        if agent.status == AgentStatus.SUCCESS:
            print(f"Build target '{target}' completed successfully!")
        else:
            print(f"Build failed: {agent.error_message}")
            
            # Ask if user wants to debug
            if target == "csim":  # We only debug C simulation errors for now
                debug_choice = input("Would you like to run the debug assistant? (y/n): ")
                if debug_choice.lower() == 'y':
                    self.do_debug("")
    
    def do_debug(self, arg):
        """Debug the current component's C simulation errors [--model model_name] [--timeout seconds]"""
        # Parse arguments
        args = arg.split()
        model_name = self.default_model
        timeout = self.debug_timeout
        
        i = 0
        while i < len(args):
            if args[i] == '--model' and i + 1 < len(args):
                model_name = args[i+1]
                i += 2
            elif args[i] == '--timeout' and i + 1 < len(args):
                try:
                    timeout = int(args[i+1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid timeout value: {args[i+1]}")
                    return
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
        
        # Check for error log file
        error_log = os.path.join(component_dir, "csim.log")
        if not os.path.isfile(error_log):
            # Check for HLS project log
            error_log = os.path.join(component_dir, f"proj_{self.current_component}", "solution1", "csim", "report", "csim_result.log")
            if not os.path.isfile(error_log):
                print(f"Error: No error log file found. Run C simulation first.")
                return
        
        # Find source files
        source_files = []
        for ext in [".hpp", ".cpp", "_tb.cpp"]:
            source_file = os.path.join(component_dir, f"{self.current_component}{ext}")
            if os.path.isfile(source_file):
                source_files.append(source_file)
        
        if not source_files:
            print("Error: No source files found for debugging")
            return
        
        # Set up context for debug agent
        agent_context = {
            "args": {
                "error_log": error_log,
                "source_file": source_files,
                "model": model_name,
                "timeout": timeout
            }
        }
        
        # Add timeout parameter
        print(f"Running debug assistant for {self.current_component}... (timeout: {timeout}s)")
        print("This may take a while. Press Ctrl+C to abort if it takes too long.")
        
        try:
            agent = self.orchestrator.agents["debug_assistant"]
            result = agent.run(agent_context)
            
            if agent.status == AgentStatus.SUCCESS:
                print("Debug analysis completed.")
                print("Check the debug report for suggested fixes")
            else:
                if result.get("timeout", False):
                    print(f"Debug analysis timed out after {timeout} seconds.")
                    print("Consider using a different model or simplifying your code before trying again.")
                    retry = input("Would you like to try code generation instead? (y/n): ")
                    if retry.lower() == 'y':
                        self.do_generate(arg)
                else:
                    print(f"Debug analysis failed: {agent.error_message}")
        except KeyboardInterrupt:
            print("\nDebug operation was interrupted.")
            retry = input("Would you like to try code generation instead? (y/n): ")
            if retry.lower() == 'y':
                self.do_generate(arg)
    
    def do_generate_makefile(self, arg):
        """Generate Makefile for the current component: generate_makefile [--template template_file]"""
        # Parse arguments
        args = arg.split()
        template_file = None
        
        i = 0
        while i < len(args):
            if args[i] == '--template' and i + 1 < len(args):
                template_file = os.path.abspath(args[i+1])
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
        
        # Use default template if not specified
        if not template_file:
            default_template = os.path.join(self.current_project_dir, "scripts", "Makefile")
            if os.path.isfile(default_template):
                template_file = default_template
            else:
                print(f"Warning: Default template Makefile not found at {default_template}")
                print("Will generate a basic Makefile without template")
        elif not os.path.isfile(template_file):
            print(f"Error: Template file not found: {template_file}")
            return
        
        # Set up context for makefile generator agent
        agent_context = {
            "work_dir": component_dir,
            "component": self.current_component,
            "template_makefile": template_file
        }
        
        # Run makefile generator agent
        print(f"Generating Makefile for component {self.current_component}...")
        agent = self.orchestrator.agents["makefile_generator"]
        result = agent.run(agent_context)
        
        if agent.status == AgentStatus.SUCCESS:
            print("Makefile generation successful!")
            makefile_path = os.path.join(component_dir, "Makefile")
            print(f"Makefile created at: {makefile_path}")
        else:
            print(f"Makefile generation failed: {agent.error_message}")
    
    def do_run_workflow(self, arg):
        """Run the complete workflow on the current component"""
        if not self.workflow_file:
            print("Error: No workflow file loaded. Use 'load_workflow' first.")
            return
        
        # Check if we have a component
        if not self.current_component:
            print("Error: No component selected. Use 'set_component' first.")
            return
        
        # Add current component to context
        workflow_context = {**self.current_context}
        
        # Run the workflow
        print(f"Running complete workflow for component {self.current_component}...")
        success = self.orchestrator.run_workflow(workflow_context)
        
        if success:
            print("Workflow completed successfully!")
        else:
            print("Workflow execution failed. Check the logs for details.")
    
    def do_history(self, arg):
        """Show execution history of agents"""
        history = self.orchestrator.get_history()
        
        if not history:
            print("No execution history available")
            return
        
        print("\nExecution History:")
        print("=" * 80)
        for i, entry in enumerate(history, 1):
            print(f"Step {i}: {entry['step']} (Agent: {entry['agent']})")
            print(f"  Status: {entry['result']['status']}")
            if entry['result']['status'] == 'failed' and 'error' in entry['result']:
                print(f"  Error: {entry['result']['error']}")
            print()
    
    def do_quit(self, arg):
        """Exit the FPGA agent CLI"""
        print("Exiting FPGA Agent CLI...")
        return True
    
    # Aliases
    do_exit = do_quit
    do_q = do_quit

def main():
    """Main function for running the FPGA agent CLI"""
    parser = argparse.ArgumentParser(description="FPGA Design Agent CLI")
    parser.add_argument("--component", type=str, help="Set the current component name")
    parser.add_argument("--project_dir", type=str, help="Set the project directory")
    parser.add_argument("--workflow", type=str, help="Load a workflow configuration file")
    parser.add_argument("--model", type=str, help="Set the LLM model to use", default="gemini-2.5-pro-exp-03-25")
    parser.add_argument("--debug-timeout", type=int, help="Set debug timeout in seconds", default=300)
    parser.add_argument("--generate-timeout", type=int, help="Set code generation timeout in seconds", default=600)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
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
    
    # Start the interactive shell
    shell.cmdloop()

if __name__ == "__main__":
    main()
