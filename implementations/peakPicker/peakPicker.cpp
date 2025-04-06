/* AUTO-EDITED BY DEBUG ASSISTANT */
#include "peakPicker.hpp"

/**
 * @brief Core implementation of the peakPicker algorithm for HLS.
 *
 * Implements a sliding window peak detector. A sample is considered a peak if:
 * 1. It is the maximum value within a window of WINDOW_LENGTH samples centered around it.
 * 2. Its value exceeds the corresponding threshold value.
 *
 * This function processes streaming input data and outputs the 0-based indices of detected peaks.
 */
void peakPicker(
    hls::stream<Data_t>& xcorrStream,
    hls::stream<Data_t>& thresholdStream,
    hls::stream<Index_t>& locationStream,
    int numSamples
) {
    // Interface pragmas are usually placed here or in a separate directives file
    // #pragma HLS INTERFACE axis port=xcorrStream
    // #pragma HLS INTERFACE axis port=thresholdStream
    // #pragma HLS INTERFACE axis port=locationStream
    // #pragma HLS INTERFACE s_axilite port=numSamples bundle=control
    // #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Buffers to hold the current window of data and thresholds
    // These implement the sliding window mechanism.
    Data_t xcorrBuffer[WINDOW_LENGTH];
    Data_t thresholdBuffer[WINDOW_LENGTH];

    // Partitioning the arrays allows parallel access to elements within the
    // pipelined loop, mapping them to registers for II=1.
    #pragma HLS ARRAY_PARTITION variable=xcorrBuffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=thresholdBuffer complete dim=1

    // Initialize buffers (optional, but good practice for simulation)
    // Can be skipped if the initial state doesn't affect the first valid output
    // Using an unrolled loop for initialization
    init_loop:
    for (int i = 0; i < WINDOW_LENGTH; ++i) {
        #pragma HLS UNROLL
        xcorrBuffer[i] = 0;
        thresholdBuffer[i] = 0;
    }

    // Main processing loop iterates through all input samples
    // Apply PIPELINE directive for high throughput (initiation interval II=1)
    main_loop:
    for (int i = 0; i < numSamples; ++i) {
        #pragma HLS PIPELINE II=1

        // 1. Shift Buffers: Make space for the new sample at index 0
        // Shift existing elements towards the end of the buffer
        shift_loop:
        for (int k = WINDOW_LENGTH - 1; k > 0; --k) {
            #pragma HLS UNROLL // Unroll this small loop for efficiency
            xcorrBuffer[k] = xcorrBuffer[k - 1];
            thresholdBuffer[k] = thresholdBuffer[k - 1];
        }

        // 2. Read New Samples: Read from input streams and place at the start (index 0)
        // Ensure streams are not empty before reading (HLS streams block if empty)
        Data_t newXcorrSample = xcorrStream.read();
        Data_t newThresholdSample = thresholdStream.read();
        xcorrBuffer[0] = newXcorrSample;
        thresholdBuffer[0] = newThresholdSample;

        // 3. Peak Detection Logic: Start after the window is filled
        // The first potential peak can be checked when i = WINDOW_LENGTH - 1
        // At this point, the sample corresponding to the middle of the *first full window*
        // is located at xcorrBuffer[MIDDLE_LOCATION]. This sample was originally read
        // at iteration i = (WINDOW_LENGTH - 1) - MIDDLE_LOCATION = i - MIDDLE_LOCATION.
        if (i >= WINDOW_LENGTH - 1) {
            // Get the sample and threshold at the center of the current window
            // This sample corresponds to the one read 'MIDDLE_LOCATION' iterations ago.
            Data_t middleSample = xcorrBuffer[MIDDLE_LOCATION];
            Data_t middleThreshold = thresholdBuffer[MIDDLE_LOCATION];

            // Check if the middle sample is the maximum in the window
            bool isPeak = true; // Assume it's a peak initially
            compare_loop:
            for (int k = 0; k < WINDOW_LENGTH; ++k) {
                 #pragma HLS UNROLL // Unroll comparison loop for parallelism
                // Check if any other sample in the window is strictly greater
                // Note: MATLAB's `findpeaks` often uses >= (non-strictly greater),
                // but the reference comment suggests middleSample >= all others.
                // This implementation finds peaks strictly greater than neighbors.
                // If equality is allowed (plateaus), the condition might need adjustment.
                // The current logic: middleSample must be >= all other elements.
                if (k != MIDDLE_LOCATION && xcorrBuffer[k] > middleSample) {
                    isPeak = false;
                    // No need to 'break' here if the loop is fully unrolled,
                    // as all comparisons happen in parallel. But break is harmless
                    // in C simulation and might help logic synthesis slightly.
                    break;
                }
            }

            // Check if the peak condition is met (local maximum AND above threshold)
            if (isPeak && (middleSample > middleThreshold)) {
                // Calculate the 0-based index of the peak in the original input stream
                // The sample currently at xcorrBuffer[MIDDLE_LOCATION] was read
                // MIDDLE_LOCATION iterations ago. Its original index was i - MIDDLE_LOCATION.
                Index_t peakLocation = i - MIDDLE_LOCATION;

                // Write the detected peak location (0-based index) to the output stream
                locationStream.write(peakLocation);
            }
        }
    } // end main_loop
}