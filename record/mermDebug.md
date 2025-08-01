# HLS Debug Automation With LLM Analysis

I've created a Mermaid flowchart that visually represents how the debug assistant uses LLMs (particularly Gemini) to automatically analyze and fix HLS simulation errors.

## Mermaid Graph

```mermaid
graph TD
    subgraph Inputs
        A[Error Log] -->|read_file| C
        B[HLS C++ Source Files] -->|read_file| D
    end

    subgraph Processing
        C[Extract Error Information] --> E
        D[Parse Source Code] --> E
        E[Create Debug Prompt] --> F
    end

    subgraph LLM_Analysis
        F[Query LLM API] -->|model selection| G{Select Model}
        G -->|gemini-2.0-pro-exp| H[Gemini API]
        G -->|gpt-4/gpt-3.5-turbo| I[OpenAI API]
        G -->|claude-sonnet| J[Claude API]
        H --> K[LLM Analysis Response]
        I --> K
        J --> K
    end

    subgraph Outputs
        K --> L[Generate Debug Report]
        K --> M[Parse Code Corrections]
        
        L --> N[Save Markdown Report]
        M --> O[Apply Code Fixes]
        O -->|user confirmation| P[Edit Source Files]
    end

    style H fill:#34A853,stroke:#34A853,color:white
    style K fill:#F9AB00,stroke:#F9AB00,color:white
    style P fill:#4285F4,stroke:#4285F4,color:white
    style N fill:#4285F4,stroke:#4285F4,color:white
```

## Explanation of the Graph

The flowchart illustrates the end-to-end process of the HLS debug assistant:

1. **Inputs Section**:
   - The process begins with two key inputs: the HLS C++ simulation error log and the source code files
   - These files are read into the system using the `read_file` function

2. **Processing Section**:
   - Error information is extracted from the log file using pattern matching (`extract_error_information`)
   - Source code is parsed and organized
   - These elements are combined to create a comprehensive debug prompt for the LLM that includes both the errors and code context

3. **LLM Analysis Section**:
   - The debug prompt is sent to an LLM API based on the model selection
   - The system supports three LLM services (highlighted in the graph):
     - Google's Gemini models (primary, highlighted in green)
     - OpenAI's GPT models
     - Anthropic's Claude models
   - The LLM processes the prompt and returns an analysis with suggested fixes

4. **Outputs Section**:
   - The LLM response is processed in two ways:
     - A comprehensive debug report is generated and saved as a Markdown file
     - Code corrections are parsed from the LLM response
   - The user can optionally apply the suggested fixes to the source files after confirmation

The key advantage of this approach is automation of the debugging process through AI assistance, particularly leveraging Gemini's capabilities to understand HLS code contexts and simulation errors, then provide targeted fixes with explanations of the underlying issues.