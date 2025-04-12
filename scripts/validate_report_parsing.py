#!/usr/bin/env python3

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("report_validator")

# Add the scripts directory to the path so we can import the agent classes
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import agent framework components
from agent_framework import DocumentationAgent

def validate_report_parsing(component_dir):
    """Validate the parsing of HLS implementation and latency reports."""
    if not os.path.exists(component_dir):
        logger.error(f"Component directory not found: {component_dir}")
        return False
    
    # Create a temporary documentation agent for testing
    doc_agent = DocumentationAgent("test_doc_agent", "Test documentation agent")
    
    # Test report parsing
    try:
        logger.info(f"Analyzing reports in {component_dir}")
        metrics = doc_agent.analyze_reports(component_dir)
        
        if "error" in metrics:
            logger.error(f"Error in report analysis: {metrics['error']}")
            return False
        
        # Check if any metrics were found
        resource_count = sum(len(impl) for impl in metrics.get("resources", {}).values())
        timing_count = sum(len(impl) for impl in metrics.get("timing", {}).values())
        latency_count = len(metrics.get("latency", {}))
        
        logger.info(f"Found resource metrics: {resource_count} entries")
        logger.info(f"Found timing metrics: {timing_count} entries")
        logger.info(f"Found latency metrics: {latency_count} entries")
        
        # Check for specific expected fields
        if "resources" in metrics and metrics["resources"]:
            first_impl = next(iter(metrics["resources"]))
            first_res = metrics["resources"][first_impl]
            logger.info(f"Resource metrics example ({first_impl}): {first_res}")
        
        if "timing" in metrics and metrics["timing"]:
            first_impl = next(iter(metrics["timing"]))
            first_timing = metrics["timing"][first_impl]
            logger.info(f"Timing metrics example ({first_impl}): {first_timing}")
            
            # Verify MHz conversion
            for key, value in first_timing.items():
                if key.endswith("_MHz"):
                    logger.info(f"  Found MHz conversion: {key} = {value}")
        
        if "latency" in metrics and metrics["latency"]:
            first_impl = next(iter(metrics["latency"]))
            first_latency = metrics["latency"][first_impl]
            logger.info(f"Latency metrics example ({first_impl}): {first_latency}")
        
        # Check if markdown report was generated
        md_report_path = os.path.join(component_dir, "performance_metrics.md")
        if os.path.exists(md_report_path):
            logger.info(f"Markdown performance report generated at: {md_report_path}")
            with open(md_report_path, 'r') as f:
                report_content = f.read()
            logger.info(f"Report length: {len(report_content)} characters")
        else:
            logger.warning(f"No markdown report found at: {md_report_path}")
        
        # Verify data extraction for documentation
        test_prompt = doc_agent._create_documentation_prompt(
            {"workflow_summary": {}},
            metrics,
            component_dir,
            {"component": os.path.basename(component_dir)},
            ["readme"],
            None
        )
        
        logger.info(f"Generated documentation prompt length: {len(test_prompt)} characters")
        
        # Extract performance metrics section from the prompt
        import re
        perf_section = re.search(r"## Performance Metrics\s*\n(.*?)(?:\n##|\Z)", test_prompt, re.DOTALL)
        if perf_section:
            logger.info("Performance metrics section found in documentation prompt")
            performance_text = perf_section.group(1)
            logger.info(f"Performance section length: {len(performance_text)} characters")
        else:
            logger.warning("Performance metrics section not found in documentation prompt")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating report parsing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate HLS report parsing and documentation generation")
    parser.add_argument("--component_dir", type=str, required=True, 
                        help="Path to component directory containing HLS reports")
    
    args = parser.parse_args()
    
    success = validate_report_parsing(args.component_dir)
    
    if success:
        logger.info("Validation completed successfully")
        sys.exit(0)
    else:
        logger.error("Validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
