#ifndef PEAK_PICKER_HPP
#define PEAK_PICKER_HPP

#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_vector.h> // Required for hls::vector

//--------------------------------------------------------------------------
// Constants and Parameters
//--------------------------------------------------------------------------

// Fixed-point precision for correlation and threshold values
constexpr int DATA_W = 20; // Total width
constexpr int DATA_I = 1; // Integer width

// Number of parallel PSS correlation sequences
constexpr int NUM_SEQUENCES = 1;

// Sliding window length (must be odd)
constexpr int WINDOW_LENGTH = 11;
static_assert(WINDOW_LENGTH % 2 != 0, "WINDOW_LENGTH must be odd");

// Middle location offset within the window (0-based index)
constexpr int MIDDLE_LOCATION = WINDOW_LENGTH / 2;

// Maximum expected number of peaks (for sizing internal buffers if needed,
// though stream output avoids large output buffers in the core function)
// This is more relevant for non-streaming outputs or internal logic.
// constexpr int MAX_PEAKS = 100; // Example if needed

//--------------------------------------------------------------------------
// Type Definitions
//--------------------------------------------------------------------------

// Fixed-point type for input data (correlation magnitude squared, threshold)
typedef ap_fixed<DATA_W, DATA_I> Data_t;

// Vector type to hold samples from all sequences at a single time index
// Useful for streaming multiple sequences concurrently
typedef hls::vector<Data_t, NUM_SEQUENCES> InputSample_t;

// Type for output peak locations (indices)
typedef int Index_t;

//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------

/**
 * @brief Identifies peaks in PSS correlation data using a sliding window.
 *
 * @param xcorrStream       Input stream of PSS correlation magnitude squared values.
 *                          Each element is an hls::vector containing values for all sequences
 *                          at a given time index.
 * @param thresholdStream   Input stream of threshold values, corresponding element-wise
 *                          to xcorrStream.
 * @param locationsStream   Output stream for detected peak locations (0-based indices).
 * @param dataLength        Total number of samples per sequence in the input streams.
 */
void peakPicker(
    hls::stream<InputSample_t>& xcorrStream,
    hls::stream<InputSample_t>& thresholdStream,
    hls::stream<Index_t>& locationsStream,
    int dataLength
);

#endif // PEAK_PICKER_HPP
