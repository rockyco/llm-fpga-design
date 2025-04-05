/* AUTO-EDITED BY DEBUG ASSISTANT */
#include "peakPicker.hpp"

/**
 * @brief Core implementation of the peak picker algorithm.
 *
 * This function implements the sliding window peak detection logic described
 * in the MATLAB reference (`peakPicker.m`). It identifies local maxima in
 * the `xcorrStream` that are also above the corresponding `thresholdStream` value.
 *
 * Optimizations:
 * - Loop Pipelining (II=1): Enables processing one sample per clock cycle after initial latency.
 * - Array Partitioning: Allows parallel access to window buffer elements for faster comparison.
 */
void peakPicker(
    XcorrStream&    xcorrStream,
    ThresholdStream& thresholdStream,
    int             numSamples,
    LocationStream& locationStream
) {
    // --- HLS Directives ---
    // Apply DATAFLOW if this function is part of a larger pipelined design.
    // For standalone use, PIPELINE on the main loop is often sufficient.
    // #pragma HLS DATAFLOW

    // Window buffers to hold the current sliding window data
    DataType xcorrWindow[WINDOW_LENGTH];
    DataType thresholdWindow[WINDOW_LENGTH];

    // --- HLS Directives for Window Buffers ---
    // Partitioning the arrays allows parallel access during the max check.
    // 'complete' partitioning creates individual registers for each element.
    #pragma HLS ARRAY_PARTITION variable=xcorrWindow complete dim=1
    #pragma HLS ARRAY_PARTITION variable=thresholdWindow complete dim=1

    // Initialize window buffers (optional, but good practice)
    // This loop will be unrolled by HLS.
    INIT_WINDOW_LOOP: for (int i = 0; i < WINDOW_LENGTH; ++i) {
        #pragma HLS UNROLL
        xcorrWindow[i] = 0;
        thresholdWindow[i] = 0;
    }

    // --- Main Processing Loop ---
    // Iterate through all input samples.
    PROCESS_SAMPLES_LOOP: for (int i = 0; i < numSamples; ++i) {
        // --- HLS Directive for Pipelining ---
        // Target an initiation interval (II) of 1 cycle.
        #pragma HLS PIPELINE II=1

        // 1. Shift window buffers
        // Shift existing elements to make space for the new one.
        // This implements the buffer shift logic from MATLAB.
        SHIFT_WINDOW_LOOP: for (int j = WINDOW_LENGTH - 1; j > 0; --j) {
            // No need for explicit UNROLL here, PIPELINE handles sequential dependencies.
            xcorrWindow[j] = xcorrWindow[j - 1];
            thresholdWindow[j] = thresholdWindow[j - 1];
        }

        // 2. Read new samples from input streams
        DataType currentXcorr;
        DataType currentThreshold;
        xcorrStream >> currentXcorr;
        thresholdStream >> currentThreshold;

        // 3. Insert new samples into the window buffers at the 'newest' position (index 0)
        xcorrWindow[0] = currentXcorr;
        thresholdWindow[0] = currentThreshold;

        // 4. Check for peak condition after the window is filled
        // The check is valid only when at least WINDOW_LENGTH samples have been processed.
        // The candidate peak corresponds to the sample currently at the middle location.
        if (i >= WINDOW_LENGTH - 1) {
            // Get the middle sample and its corresponding threshold
            DataType midSample = xcorrWindow[MIDDLE_LOCATION];
            DataType midThreshold = thresholdWindow[MIDDLE_LOCATION];

            // Check if the middle sample is the maximum in the window
            bool isMax = true;
            CHECK_MAX_LOOP: for (int k = 0; k < WINDOW_LENGTH; ++k) {
                 // No need for explicit UNROLL here, partitioning enables parallel checks within II=1.
                // Note: MATLAB uses <= 0 after subtraction. Here we compare directly.
                // Ensure comparison matches MATLAB: middle sample must be >= all others.
                // This check correctly implements that: if any other element is strictly greater,
                // midSample is not the maximum.
                if (xcorrWindow[k] > midSample) {
                    isMax = false;
                    break; // Exit early if a larger element is found
                }
            }

            // Peak condition: Middle sample is local maximum AND exceeds threshold
            if (isMax && (midSample > midThreshold)) {
                // Calculate the index of the detected peak.
                // The sample at the middle of the *current* window (index `MIDDLE_LOCATION`)
                // corresponds to the input sample from `MIDDLE_LOCATION` iterations ago.
                // This gives the 0-based index.
                LocationType peakLocation = i - MIDDLE_LOCATION;

                // Write the detected peak location to the output stream
                locationStream << peakLocation;
            }
        }
    } // end PROCESS_SAMPLES_LOOP
}