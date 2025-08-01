# HLS Performance Optimization Guidelines

## Task Description
You are tasked with optimizing an existing HLS C++ implementation to improve performance, reduce resource utilization, or achieve better timing. Your goal is to maintain functional correctness while enhancing design metrics.

## Source Files
The following HLS C++ source files are being provided:

{{SOURCE_FILES}}

## Performance Metrics
Current performance metrics of the design:

{{PERFORMANCE_METRICS}}

## Optimization Goals
- Primary goal: {{PRIMARY_GOAL}} (e.g., "Reduce latency by at least 30%")
- Secondary goal: {{SECONDARY_GOAL}} (e.g., "Maintain or reduce resource utilization")

## Optimization Strategy

Please follow this structured approach:

1. **Design Analysis**
   - Analyze the algorithm structure and computational patterns
   - Identify performance bottlenecks in the current implementation
   - Map data dependencies and memory access patterns
   - Recognize rate-limiting operations or loops

2. **Optimization Techniques**
   
   Consider the following optimization categories:
   
   **Loop Optimizations:**
   - Pipeline loops to improve throughput (PIPELINE pragma)
   - Unroll loops to exploit parallelism (UNROLL pragma)
   - Merge loops to reduce overhead
   - Partition loops to enable better scheduling
   
   **Memory Optimizations:**
   - Array partitioning (ARRAY_PARTITION pragma)
   - Memory reshaping for better access patterns
   - Double buffering for overlapped computation
   - Streaming interfaces for sequential data (hls::stream)
   
   **Data Type Optimizations:**
   - Optimize bit widths using ap_fixed/ap_int
   - Convert floating-point to fixed-point where appropriate
   - Simplify complex operations with lookup tables or approximations
   
   **Function-Level Optimizations:**
   - Inline small functions to reduce function call overhead
   - Dataflow optimization for task-level pipelining
   - Function parallelism with multiple instances
   
   **Interface Optimizations:**
   - Optimize interface protocols (AXI4, AXI-Lite, AXI-Stream)
   - Burst transfers for efficient data movement
   - Register slicing for timing improvement

3. **Implementation Plan**
   - Prioritize optimizations based on impact vs. effort
   - Plan incremental changes that can be verified individually
   - Consider trade-offs between different metrics (latency vs. area)

## Response Format

Please structure your response as follows:

### 1. Design Analysis
A summary of your analysis of the current implementation, identifying bottlenecks and opportunities.

### 2. Recommended Optimizations
For each file requiring changes:
- The file name
- Description of proposed optimizations
- Code snippets showing the modifications with added pragmas or code changes
- Expected impact of each optimization

### 3. Implementation Priority
A prioritized list of optimizations, explaining which should be implemented first.

### 4. Expected Outcomes
Predictions about the performance improvements that could be achieved.

### 5. Verification Plan
Suggestions for verifying that the optimizations maintain functional correctness.

## Additional Guidelines
- Focus on HLS-specific optimizations, not general C++ performance improvements
- Explain the reasoning behind each optimization
- Consider Xilinx/Intel FPGA architecture specifics when relevant
- Indicate any potential risks or trade-offs for each optimization
- When multiple approaches could work, explain the pros and cons of each
