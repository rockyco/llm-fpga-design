# Debug Report

## Error Information
```
ERR: [SIM 100] 'csim_design' failed: nonzero return value.
Test FAILED: Found 3 mismatches (comparing HLS 0-based output with adjusted 0-based reference).
```

## LLM Analysis and Suggestions (gemini-2.5-pro-exp-03-25)
Okay, let's analyze the code and the error message.

## Analysis of the Issue

1.  **Error Message:** The core error `csim_design failed: nonzero return value` indicates that the testbench (`peakPicker_tb.cpp`) exited with an error status. The testbench itself reports `Found 3 mismatches (comparing HLS 0-based output with adjusted 0-based reference)`. This means the HLS implementation produced different peak locations than expected based on the reference file (`locations_3_ref.txt`, adjusted for 0-based indexing).

2.  **Code Examination (`peakPicker.cpp`):**
    *   **Algorithm:** The code implements a standard sliding window peak detector. It checks if the middle element of the window is both above a threshold and the maximum value within that window.
    *   **Indexing:** The calculation `candidate_location = i - MIDDLE_LOCATION` correctly determines the original 0-based index of the element currently residing in the middle (`window_buf[MIDDLE_LOCATION]`) of the sliding window, given that `i` is the index of the *newest* element just added to the window at `window_buf[WINDOW_LENGTH - 1]`.
    *   **Window Management:** The code uses arrays (`window_buf`, `threshold_buf`) partitioned completely, acting as shift registers. The `SHIFT_WINDOW_LOOP` correctly shifts elements down. New elements are read into the last position (`WINDOW_LENGTH - 1`).
    *   **Initialization (`INIT_FILL_LOOP`):** This loop runs from `i = 0` to `WINDOW_LENGTH - 2`. It reads `WINDOW_LENGTH - 1` elements and fills the buffers from index `0` up to `WINDOW_LENGTH - 2`.
    *   **Main Loop (`MAIN_PROC_LOOP`):** This loop starts at `i = WINDOW_LENGTH - 1`.
    *   **The Bug:** The critical issue lies in the interaction between the initial fill and the first iteration of the main loop.
        *   After `INIT_FILL_LOOP`, `window_buf[0]` to `window_buf[WINDOW_LENGTH - 2]` contain the first `WINDOW_LENGTH - 1` input values. `window_buf[WINDOW_LENGTH - 1]` is uninitialized.
        *   In the *first* iteration of `MAIN_PROC_LOOP` (`i = WINDOW_LENGTH - 1`):
            *   The `SHIFT_WINDOW_LOOP` executes *first*. It attempts to shift data: `window_buf[k] = window_buf[k + 1]`. When `k = WINDOW_LENGTH - 2`, this line becomes `window_buf[WINDOW_LENGTH - 2] = window_buf[WINDOW_LENGTH - 1]`. This reads the uninitialized value from `window_buf[WINDOW_LENGTH - 1]` and propagates it into the buffer at index `WINDOW_LENGTH - 2`.
            *   *Then*, the code reads the next input value (`xcorr_strm.read()`, corresponding to original index `i = WINDOW_LENGTH - 1`) and places it into `window_buf[WINDOW_LENGTH - 1]`.
            *   The subsequent peak detection logic operates on a buffer that was corrupted by the uninitialized read during the shift. This likely causes incorrect peak detection results, especially near the beginning of the data stream, leading to the mismatches reported by the testbench.

3.  **Correction Strategy:** The most straightforward way to fix this while maintaining the `Shift -> Read -> Check` structure within the main loop is to ensure the *entire* window buffer is initialized with valid data *before* the first shift operation occurs. This can be done by reading one element ahead before the main loop starts and managing the reading process carefully within the loop.

    *   Fill the first `WINDOW_LENGTH - 1` elements as before.
    *   Read the element corresponding to index `WINDOW_LENGTH - 1` into temporary variables *before* the main loop.
    *   In the main loop (`i` from `WINDOW_LENGTH - 1` to `data_length - 1`):
        *   Perform the shift.
        *   Place the value read *before* this iteration (stored in the temporary variables) into `window_buf[WINDOW_LENGTH - 1]`.
        *   Read the *next* input value (for the *next* iteration) into the temporary variables.
        *   Perform the peak check.

## COMPLETE CORRECTED SOURCE CODE:

**File: `peakPicker.hpp`**

```cpp
/* AUTO-EDITED BY DEBUG ASSISTANT */
#ifndef PEAK_PICKER_HPP
#define PEAK_PICKER_HPP

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h> // For ap_uint

//--------------------------------------------------------------------------
// Constants and Parameters
//--------------------------------------------------------------------------

// Fixed-point type configuration (Adjust W, I based on required precision/range)
// W = Total width, I = Integer width
constexpr int DATA_W = 32;
constexpr int DATA_I = 16;
constexpr int THRESH_W = 32;
constexpr int THRESH_I = 16;

// Window parameters (Must match MATLAB implementation)
constexpr int WINDOW_LENGTH = 11;
constexpr int MIDDLE_LOCATION = WINDOW_LENGTH / 2; // 0-based index for C++

// Maximum expected data length (for sizing internal buffers if needed, affects resource usage)
// Set appropriately based on expected input size. If unknown, make it large enough.
// For purely streaming designs without needing the full input buffered, this might not be strictly necessary
// but can be useful for loop bounds in some implementations. Let's assume a reasonable max for now.
constexpr int MAX_DATA_LENGTH = 4096; // Example maximum length

// Stream buffer depth (Adjust based on producer/consumer rates and simulation needs)
constexpr int STREAM_BUFFER_DEPTH = 2;

//--------------------------------------------------------------------------
// Type Definitions
//--------------------------------------------------------------------------

// Fixed-point types for data and threshold
// Using ap_ufixed as correlation magnitude squared and threshold are likely non-negative
// If negative values are possible, switch back to ap_fixed
using Data_t = ap_ufixed<DATA_W, DATA_I, AP_RND_CONV, AP_SAT>; // Example: Unsigned 32-bit, 16 integer bits
using Threshold_t = ap_ufixed<THRESH_W, THRESH_I, AP_RND_CONV, AP_SAT>; // Example: Same format as data

// Type for output indices (Ensure it's wide enough for MAX_DATA_LENGTH)
// Using int is common, but ap_uint can be more specific for hardware
using Index_t = int; // Standard integer type for indices

// Stream types
using DataStream_t = hls::stream<Data_t>;
using ThresholdStream_t = hls::stream<Threshold_t>;
using IndexStream_t = hls::stream<Index_t>;

//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------

/**
 * @brief Implements the peak picker algorithm for SSB detection.
 *
 * Finds indices where the PSS correlation magnitude squared ('xcorr')
 * is a local maximum within a sliding window and exceeds a threshold.
 * Outputs 0-based indices.
 *
 * @param xcorr_strm       Input stream of PSS correlation magnitude squared values.
 * @param threshold_strm   Input stream of threshold values (one per xcorr value).
 * @param locations_strm   Output stream of detected peak locations (0-based indices).
 * @param data_length      The total number of samples in the input streams.
 */
void peakPicker(
    DataStream_t& xcorr_strm,
    ThresholdStream_t& threshold_strm,
    IndexStream_t& locations_strm,
    int data_length
);

#endif // PEAK_PICKER_HPP
```

**File: `peakPicker.cpp`**

```cpp
/* AUTO-EDITED BY DEBUG ASSISTANT */
#include "peakPicker.hpp"
#include <cassert> // Include for assert

/**
 * @brief Core peak picking logic using a sliding window.
 *
 * This function implements the peak picking algorithm described in the MATLAB reference.
 * It uses shift registers (implemented as arrays) to efficiently manage the sliding window.
 * It outputs 0-based indices.
 *
 * @param xcorr_strm       Input stream of PSS correlation magnitude squared values.
 * @param threshold_strm   Input stream of threshold values (one per xcorr value).
 * @param locations_strm   Output stream of detected peak locations (0-based indices).
 * @param data_length      The total number of samples in the input streams.
 */
void peakPicker(
    DataStream_t& xcorr_strm,
    ThresholdStream_t& threshold_strm,
    IndexStream_t& locations_strm,
    int data_length
) {
    // --- HLS Interface Pragmas ---
    // Define stream interfaces as AXI-Stream
    #pragma HLS INTERFACE axis port=xcorr_strm      bundle=INPUT_STREAM
    #pragma HLS INTERFACE axis port=threshold_strm  bundle=INPUT_STREAM
    #pragma HLS INTERFACE axis port=locations_strm  bundle=OUTPUT_STREAM

    // Define data_length and function control as AXI-Lite
    #pragma HLS INTERFACE s_axilite port=data_length bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=return       bundle=CONTROL

    // --- Input Assertions (for simulation/debugging) ---
    // Ensure window length is odd and positive
    static_assert(WINDOW_LENGTH > 0, "WINDOW_LENGTH must be positive.");
    static_assert(WINDOW_LENGTH % 2 != 0, "WINDOW_LENGTH must be odd.");
    // Basic check on data length (can be enabled for debug)
    // Use assert for runtime checks during simulation if needed
    // assert(data_length >= WINDOW_LENGTH && "Data length must be at least window length for any peak detection.");
    // assert(data_length <= MAX_DATA_LENGTH && "Data length exceeds maximum specified.");

    // Handle cases where data_length is too short for any peak detection
    if (data_length < WINDOW_LENGTH) {
        // Consume inputs to prevent deadlock, but produce no output
        // Note: This consumption loop might not be synthesizable if data_length
        // is determined at runtime in hardware without a clear upper bound known
        // at synthesis time. For pure C simulation, it's fine.
        // AXI-Stream interfaces usually handle termination via TLAST,
        // so explicit consumption might not be needed in hardware if properly configured.
        // Let's assume for simulation we need to consume remaining data if any.
        // However, the testbench populates exactly data_length items, so this
        // loop might not be strictly necessary if data_length is accurate.
        // If the function is called with data_length < WINDOW_LENGTH, it should just return.
        return;
    }


    // --- Internal Buffers (Shift Registers for Sliding Window) ---
    // Use arrays to implement shift registers for the sliding window.
    Data_t      window_buf[WINDOW_LENGTH];
    Threshold_t threshold_buf[WINDOW_LENGTH];

    // --- HLS Optimization Pragmas for Buffers ---
    // Partition arrays completely to allow parallel access for max check within II=1
    #pragma HLS ARRAY_PARTITION variable=window_buf    complete dim=1
    #pragma HLS ARRAY_PARTITION variable=threshold_buf complete dim=1


    // --- Initial Window Fill (Partial) ---
    // Fill the first WINDOW_LENGTH - 1 elements of the shift registers.
    // These will occupy indices 0 to WINDOW_LENGTH - 2 initially.
    INIT_FILL_LOOP: for (int i = 0; i < WINDOW_LENGTH - 1; ++i) {
        // This loop can be pipelined if needed, but it's sequential fill.
        #pragma HLS PIPELINE II=1
        window_buf[i] = xcorr_strm.read();
        threshold_buf[i] = threshold_strm.read();
    }

    // --- Variables to hold the next value to be placed in the window ---
    // Read the value corresponding to index WINDOW_LENGTH - 1 ahead of time.
    Data_t      next_xcorr_val = xcorr_strm.read();
    Threshold_t next_thresh_val = threshold_strm.read();


    // --- Main Processing Loop (Sliding Window) ---
    // Iterate from the point the window is first full until the end.
    // 'i' represents the 0-based index of the *newest* element conceptually
    // entering the window's end position in this iteration.
    MAIN_PROC_LOOP: for (int i = WINDOW_LENGTH - 1; i < data_length; ++i) {
        #pragma HLS PIPELINE II=1

        // --- Shift Window Buffers ---
        // Shift existing elements down towards index 0.
        // The value previously at window_buf[WINDOW_LENGTH-1] is shifted into [WINDOW_LENGTH-2], etc.
        // The value previously at window_buf[0] is shifted out.
        SHIFT_WINDOW_LOOP: for (int k = 0; k < WINDOW_LENGTH - 1; ++k) {
           #pragma HLS UNROLL // Explicit unroll helps ensure parallel shift
           window_buf[k] = window_buf[k + 1];
           threshold_buf[k] = threshold_buf[k + 1];
        }

        // --- Place the PREVIOUSLY read data into the end of the window ---
        // This completes the window state for the current check.
        window_buf[WINDOW_LENGTH - 1] = next_xcorr_val;
        threshold_buf[WINDOW_LENGTH - 1] = next_thresh_val;

        // --- Read the NEXT data point for the *next* iteration ---
        // Only read if we are not processing the very last input element 'i'
        // If i == data_length - 1, we don't need to read ahead anymore.
        if (i < data_length - 1) {
            next_xcorr_val = xcorr_strm.read();
            next_thresh_val = threshold_strm.read();
        }
        // If i == data_length - 1, the values in next_xcorr_val/next_thresh_val
        // are the ones just placed into the buffer, but they won't be used again.


        // --- Peak Detection Logic ---
        // The window buffer window_buf now contains data corresponding to original
        // input indices from i - (WINDOW_LENGTH - 1) up to i.
        Data_t      middle_val = window_buf[MIDDLE_LOCATION];
        Threshold_t middle_threshold = threshold_buf[MIDDLE_LOCATION];

        // Calculate the original 0-based index corresponding to the middle element
        // Index 'i' is the original index of the *newest* element in the window (at buf[W-1]).
        // The middle element is MIDDLE_LOCATION positions away from the end of the buffer.
        // Index calculation: i - ((WINDOW_LENGTH - 1) - MIDDLE_LOCATION)
        // Simplified: i - WINDOW_LENGTH + 1 + MIDDLE_LOCATION
        // Example W=11, M=5: i - 11 + 1 + 5 = i - 5.
        // The original code's calculation was correct:
        Index_t candidate_location = i - MIDDLE_LOCATION;

        // 1. Check if the middle value meets or exceeds its threshold
        bool threshold_passed = (middle_val >= middle_threshold);

        // 2. Check if the middle value is the maximum in the current window
        bool is_maximum = true;
        // This loop must complete within one clock cycle for II=1.
        // Requires window_buf to be fully partitioned.
        CHECK_MAX_LOOP: for (int k = 0; k < WINDOW_LENGTH; ++k) {
           #pragma HLS UNROLL // Explicit unroll helps ensure parallel check
           // Check if any other element in the window is strictly greater
           // If middle_val is equal to another element, it can still be the maximum (plateau)
           if (middle_val < window_buf[k]) {
               is_maximum = false;
               break; // Exit early if a larger value is found
           }
        }

        // --- Output Peak Location ---
        // If both conditions are met, write the 0-based location to the output stream
        if (threshold_passed && is_maximum) {
            locations_strm.write(candidate_location);
        }
    } // End of main processing loop
}
```

**File: `peakPicker_tb.cpp`**

```cpp
/* AUTO-EDITED BY DEBUG ASSISTANT */
#include "peakPicker.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>   // For std::abs
#include <limits>  // For numeric_limits
#include <iomanip> // For std::setprecision
#include <algorithm> // For std::min, std::max

// Define input/output file names (adjust sequence number '#')
const std::string XCORR_IN_FILE = "pssCorrMagSq_3_in.txt";
const std::string THRESH_IN_FILE = "threshold_in.txt";
const std::string LOCS_REF_FILE = "locations_3_ref.txt";
const std::string LOCS_OUT_FILE = "peakLocs_out.txt"; // Matches MATLAB tb output

// Function to read floating-point data from a file into a vector
bool readDataFile(const std::string& filename, std::vector<double>& data) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    double val;
    while (infile >> val) {
        data.push_back(val);
    }
    // Check for read errors (e.g., non-numeric data) after the loop
    if (infile.bad() || (!infile.eof() && infile.fail())) {
        std::cerr << "Error: Failed while reading file: " << filename << std::endl;
        infile.close();
        return false;
    }
    infile.close();
    std::cout << "Read " << data.size() << " values from " << filename << std::endl;
    return true;
}

// Function to read integer data from a file into a vector
bool readIntFile(const std::string& filename, std::vector<Index_t>& data) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    Index_t val;
    while (infile >> val) {
        data.push_back(val);
    }
     // Check for read errors (e.g., non-numeric data) after the loop
    if (infile.bad() || (!infile.eof() && infile.fail())) {
        std::cerr << "Error: Failed while reading file: " << filename << std::endl;
        infile.close();
        return false;
    }
    infile.close();
    std::cout << "Read " << data.size() << " values from " << filename << std::endl;
    return true;
}

// Function to write integer data to a file
bool writeIntFile(const std::string& filename, const std::vector<Index_t>& data) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }
    // Set precision for consistency if needed, though these are ints
    // outfile << std::fixed << std::setprecision(0);
    for (const auto& val : data) {
        outfile << val << std::endl; // Write one value per line
    }
     if (outfile.bad()) {
        std::cerr << "Error: Failed while writing file: " << filename << std::endl;
        outfile.close();
        return false;
    }
    outfile.close();
    std::cout << "Wrote " << data.size() << " values to " << filename << std::endl;
    return true;
}


int main() {
    std::cout << "--- Peak Picker HLS C++ Testbench ---" << std::endl;

    // --- Read Input Data ---
    std::vector<double> xcorr_double;
    std::vector<double> threshold_double;
    if (!readDataFile(XCORR_IN_FILE, xcorr_double)) return 1;
    if (!readDataFile(THRESH_IN_FILE, threshold_double)) return 1;

    // --- Read Reference Data ---
    std::vector<Index_t> locations_ref_orig; // Read original reference indices
    if (!readIntFile(LOCS_REF_FILE, locations_ref_orig)) return 1;

    // --- Basic Input Validation ---
    if (xcorr_double.size() != threshold_double.size()) {
        std::cerr << "Error: Input xcorr size (" << xcorr_double.size()
                  << ") does not match threshold size (" << threshold_double.size()
                  << ")." << std::endl;
        return 1;
    }
    if (xcorr_double.empty()) {
        std::cerr << "Error: Input data files are empty." << std::endl;
        return 1;
    }
     // Check if data length is sufficient for at least one window operation
     // The DUT now handles this internally, but warning is still useful.
     if (xcorr_double.size() < WINDOW_LENGTH) {
        std::cerr << "Warning: Input data length (" << xcorr_double.size()
                  << ") is less than WINDOW_LENGTH (" << WINDOW_LENGTH
                  << "). No peaks can be detected by HLS function." << std::endl;
        // Allow running to check empty output case if reference is also empty
    }

    int data_length = xcorr_double.size();
    std::cout << "Input data length: " << data_length << std::endl;

    // --- Prepare HLS Streams ---
    DataStream_t      xcorr_strm("xcorr_stream");
    ThresholdStream_t threshold_strm("threshold_stream");
    IndexStream_t     locations_strm("locations_stream");

    // --- Populate Input Streams ---
    std::cout << "Populating input streams..." << std::endl;
    for (int i = 0; i < data_length; ++i) {
        // Convert double to fixed-point type
        Data_t xcorr_fixed = static_cast<Data_t>(xcorr_double[i]);
        Threshold_t thresh_fixed = static_cast<Threshold_t>(threshold_double[i]);

        // Write to streams
        xcorr_strm.write(xcorr_fixed);
        threshold_strm.write(thresh_fixed);

        // Optional: Check conversion if debugging precision issues
        // if (i < 10 || i > data_length - 10) { // Print first/last few conversions
        //     std::cout << std::fixed << std::setprecision(10); // Increase precision for debug
        //     std::cout << "  Input[" << i << "]: xcorr=" << xcorr_double[i] << " -> " << xcorr_fixed.to_string(10) // Print fixed-point value
        //               << ", thresh=" << threshold_double[i] << " -> " << thresh_fixed.to_string(10) << std::endl;
        // }
    }
    std::cout << "Input streams populated." << std::endl;

    // --- Call the DUT (Device Under Test) ---
    std::cout << "Calling peakPicker HLS function..." << std::endl;
    peakPicker(xcorr_strm, threshold_strm, locations_strm, data_length);
    std::cout << "HLS function execution finished." << std::endl;

    // --- Collect Output Results ---
    std::vector<Index_t> locations_hls;
    std::cout << "Reading output stream..." << std::endl;
    while (!locations_strm.empty()) {
        locations_hls.push_back(locations_strm.read());
    }
    std::cout << "Read " << locations_hls.size() << " peak locations (0-based) from HLS implementation." << std::endl;

    // --- Write HLS Output to File ---
    if (!writeIntFile(LOCS_OUT_FILE, locations_hls)) {
        std::cerr << "Warning: Failed to write HLS output file." << std::endl;
        // Continue with verification
    }

    // --- Verification ---
    std::cout << "Comparing HLS results (0-based) with reference (assuming 1-based)..." << std::endl;
    int errors = 0;

    // Adjust reference indices from 1-based to 0-based for comparison
    std::vector<Index_t> locations_ref_0based = locations_ref_orig;
    for (Index_t& ref_idx : locations_ref_0based) {
        ref_idx -= 1; // Convert 1-based to 0-based
    }
    std::cout << "Adjusted reference indices from 1-based to 0-based for comparison." << std::endl;


    // Compare sizes first
    size_t hls_size = locations_hls.size();
    size_t ref_size = locations_ref_0based.size();
    if (hls_size != ref_size) {
        std::cout << "Mismatch: Number of peaks found differs!" << std::endl;
        std::cout << "  HLS found (0-based): " << hls_size << std::endl;
        std::cout << "  Reference expected (adjusted to 0-based): " << ref_size << std::endl;
        // Don't increment errors here yet, let the element comparison find them
    }

    // Compare element by element up to the maximum size checked
    size_t max_compare_idx = std::max(hls_size, ref_size);
    for (size_t i = 0; i < max_compare_idx; ++i) {
        bool mismatch = false;
        Index_t hls_val = (i < hls_size) ? locations_hls[i] : -1; // Use sentinel if out of bounds
        Index_t ref_val = (i < ref_size) ? locations_ref_0based[i] : -1; // Use sentinel if out of bounds
        Index_t ref_orig_val = (i < locations_ref_orig.size()) ? locations_ref_orig[i] : -1; // For reporting

        if (i >= hls_size) {
            mismatch = true;
            std::cout << "Mismatch at comparison index " << i << ":" << std::endl;
            std::cout << "  HLS Output (0-based): <MISSING>" << std::endl;
            std::cout << "  Reference  (1-based original: " << ref_orig_val << ", adjusted to 0-based: " << ref_val << ")" << std::endl;
        } else if (i >= ref_size) {
             mismatch = true;
            std::cout << "Mismatch at comparison index " << i << ":" << std::endl;
            std::cout << "  HLS Output (0-based): " << hls_val << std::endl;
            std::cout << "  Reference  (adjusted to 0-based): <MISSING>" << std::endl;
        } else if (hls_val != ref_val) {
            mismatch = true;
            std::cout << "Mismatch at comparison index " << i << ":" << std::endl;
            std::cout << "  HLS Output (0-based): " << hls_val << std::endl;
            std::cout << "  Reference  (1-based original: " << ref_orig_val << ", adjusted to 0-based: " << ref_val << ")" << std::endl;
        }

        if (mismatch) {
            errors++;
        }
    }

    // --- Final Result ---
    std::cout << "----------------------------------------" << std::endl;
    if (errors == 0) {
        std::cout << "Test PASSED: HLS output matches reference output (after adjusting reference to 0-based indexing)." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        return 0; // Success
    } else {
        // The message now matches the format from the original error log
        std::cout << "Test FAILED: Found " << errors << " mismatches (comparing HLS 0-based output with adjusted 0-based reference)." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        return 1; // Failure
    }
}
```

## Source Files
- `/home/jielei/Projects/UTS/llm-fpga-design/implementations/peakPicker/peakPicker.hpp`
- `/home/jielei/Projects/UTS/llm-fpga-design/implementations/peakPicker/peakPicker.cpp`
- `/home/jielei/Projects/UTS/llm-fpga-design/implementations/peakPicker/peakPicker_tb.cpp`

Generated on: 2025-04-05 10:14:50
