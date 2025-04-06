/* AUTO-EDITED BY DEBUG ASSISTANT */
#ifndef PEAK_PICKER_HPP
#define PEAK_PICKER_HPP

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h> // For integer types if needed for indices

//--------------------------------------------------------------------------
// Constants and Parameters
//--------------------------------------------------------------------------

// Fixed-point type configuration (Adjust W and I based on signal analysis)
constexpr int DATA_W = 32; // Total width
constexpr int DATA_I = 16; // Integer width (including sign bit)

// Sliding window configuration (Matches MATLAB reference)
constexpr int WINDOW_LENGTH = 11;
constexpr int MIDDLE_LOCATION = WINDOW_LENGTH / 2; // Index 5 (0-based)

// Index type configuration (Ensure it can hold the maximum sample index)
constexpr int INDEX_W = 32; // Width for location indices

//--------------------------------------------------------------------------
// Type Definitions
//--------------------------------------------------------------------------

// Fixed-point type for input data (xcorr and threshold)
typedef ap_fixed<DATA_W, DATA_I> Data_t;

// Type for output location indices
// Using ap_uint for non-negative indices. Use ap_int if indices can be negative.
// Or simply use 'int' if standard integer sizes are sufficient and synthesizable.
typedef ap_uint<INDEX_W> Index_t;
// typedef int Index_t; // Alternative if standard int is sufficient

//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------

/**
 * @brief Finds peaks in a cross-correlation signal using a sliding window.
 *
 * @param xcorrStream     Input stream of cross-correlation magnitude squared values.
 * @param thresholdStream Input stream of threshold values (corresponding to xcorr samples).
 * @param locationStream  Output stream for detected peak location indices (0-based).
 * @param numSamples      Total number of samples to process from the input streams.
 */
void peakPicker(
    hls::stream<Data_t>& xcorrStream,
    hls::stream<Data_t>& thresholdStream,
    hls::stream<Index_t>& locationStream,
    int numSamples
);

#endif // PEAK_PICKER_HPP