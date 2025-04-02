#include "peakPicker.hpp"

/**
 * @brief Core implementation of the peak picker algorithm for HLS.
 *
 * Implements a sliding window approach to find peaks where the center element
 * exceeds a threshold and is the maximum within the window for at least one sequence.
 */
void peakPicker(
    hls::stream<InputSample_t>& xcorrStream,
    hls::stream<InputSample_t>& thresholdStream,
    hls::stream<Index_t>& locationsStream,
    int dataLength
) {
    // --- HLS Interface Pragmas ---
    // Map streams to AXI-Stream interfaces
    #pragma HLS INTERFACE axis port=xcorrStream      bundle=INPUT_STREAM
    #pragma HLS INTERFACE axis port=thresholdStream  bundle=INPUT_STREAM
    #pragma HLS INTERFACE axis port=locationsStream  bundle=OUTPUT_STREAM

    // Map scalar arguments and control signals to AXI-Lite
    #pragma HLS INTERFACE s_axilite port=dataLength bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=return    bundle=CONTROL_BUS

    // Enable task-level pipelining (Dataflow) if applicable to the broader design context
    // For this single-loop function, PIPELINE is the primary optimization.
    // #pragma HLS DATAFLOW // Uncomment if peakPicker is part of a larger dataflow region

    // --- Local Buffers (Line Buffers for Sliding Window) ---
    // Buffer to hold the current window of correlation samples
    InputSample_t windowBuffer[WINDOW_LENGTH];
    // Buffer to hold the corresponding threshold values for the window center check
    InputSample_t thresholdBuffer[WINDOW_LENGTH];

    // --- HLS Optimization Pragmas for Buffers ---
    // Partition arrays completely to allow parallel access needed for II=1 pipeline
    #pragma HLS ARRAY_PARTITION variable=windowBuffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=thresholdBuffer complete dim=1
    // For small arrays like these, partitioning into registers is efficient.
    // If WINDOW_LENGTH were much larger (e.g., > 64 or 128), BRAM with partitioning
    // might be considered using BIND_STORAGE.

    // --- Main Processing Loop ---
    // Iterates through each sample of the input streams
    for (int i = 0; i < dataLength; ++i) {
        // Apply pipeline directive for high throughput (Initiation Interval = 1)
        #pragma HLS PIPELINE II=1

        // --- Read Inputs ---
        InputSample_t currentXcorrSample = xcorrStream.read();
        InputSample_t currentThresholdSample = thresholdStream.read();

        // --- Shift Window Buffers ---
        // Shift existing data down by one position
        SHIFT_BUFFERS_LOOP:
        for (int k = 0; k < WINDOW_LENGTH - 1; ++k) {
            #pragma HLS UNROLL // Unroll this small loop for parallel register moves
            windowBuffer[k] = windowBuffer[k + 1];
            thresholdBuffer[k] = thresholdBuffer[k + 1];
        }
        // Add the new samples to the end of the buffers
        windowBuffer[WINDOW_LENGTH - 1] = currentXcorrSample;
        thresholdBuffer[WINDOW_LENGTH - 1] = currentThresholdSample;

        // --- Peak Detection Logic ---
        // Start processing only after the window buffer is full
        if (i >= WINDOW_LENGTH - 1) {
            // Get the sample and threshold corresponding to the middle of the window
            // These were read MIDDLE_LOCATION iterations ago and are now at index MIDDLE_LOCATION
            InputSample_t middleXcorrSample = windowBuffer[MIDDLE_LOCATION];
            InputSample_t middleThreshold = thresholdBuffer[MIDDLE_LOCATION];

            // Calculate the absolute index (0-based) of the middle sample in the original sequence
            // Index i corresponds to the *end* of the current window.
            // The window covers indices [i - WINDOW_LENGTH + 1, i].
            // The middle element is at absolute index: (i - WINDOW_LENGTH + 1) + MIDDLE_LOCATION
            Index_t candidateIndexAbs = i - WINDOW_LENGTH + 1 + MIDDLE_LOCATION;

            bool peakFound = false; // Flag: set if peak condition met for ANY sequence

            // Check each sequence for peak conditions
            CHECK_SEQUENCES_LOOP:
            for (int s = 0; s < NUM_SEQUENCES; ++s) {
                #pragma HLS UNROLL // Unroll sequence check for parallelism

                // Condition 1: Middle value must be >= threshold for this sequence
                if (middleXcorrSample[s] >= middleThreshold[s]) {

                    // Condition 2: Middle value must be the maximum in the window for this sequence
                    bool isMaxInWindow = true;
                    CHECK_WINDOW_MAX_LOOP:
                    for (int k = 0; k < WINDOW_LENGTH; ++k) {
                        #pragma HLS UNROLL // Unroll window check for parallelism
                        // Compare middle element with every element in the window for sequence 's'
                        if (middleXcorrSample[s] < windowBuffer[k][s]) {
                            isMaxInWindow = false;
                            break; // No need to check further in this window for this sequence
                        }
                    } // end CHECK_WINDOW_MAX_LOOP

                    // If both conditions met for this sequence 's', a peak is found at this location.
                    // The MATLAB code outputs the location if *any* sequence meets the criteria.
                    if (isMaxInWindow) {
                        peakFound = true;
                        // Optimization: If a peak is found for any sequence,
                        // we can stop checking other sequences for this window position.
                        break; // Exit CHECK_SEQUENCES_LOOP
                    }
                } // end if (threshold check)
            } // end CHECK_SEQUENCES_LOOP

            // --- Write Output ---
            // If a peak was found for any sequence at this candidate location, write the index.
            if (peakFound) {
                locationsStream.write(candidateIndexAbs);
            }
        } // end if (window full)
    } // end main processing loop
}
