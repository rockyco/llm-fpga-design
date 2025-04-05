/* AUTO-EDITED BY DEBUG ASSISTANT */
#ifndef PEAK_PICKER_HPP
#define PEAK_PICKER_HPP

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h> // For integer types if needed, though standard int often suffices

//--------------------------------------------------------------------------
// Constants and Parameters
//--------------------------------------------------------------------------

// Window length for peak detection (must match MATLAB)
constexpr int WINDOW_LENGTH = 11;

// Middle location index (0-based) within the window
constexpr int MIDDLE_LOCATION = WINDOW_LENGTH / 2; // Integer division gives floor

// Fixed-point type definition for correlation magnitude squared and threshold
// W = Total width, I = Integer width. Adjust based on expected data range and precision.
// Example: 32 bits total, 16 integer bits (range approx +/- 32768), 16 fractional bits.
// Ensure this choice prevents overflow and provides sufficient precision.
constexpr int DATA_W = 32;
constexpr int DATA_I = 16;
typedef ap_fixed<DATA_W, DATA_I> DataType;

// Type definition for peak location indices
// Using standard int, assuming indices fit within its range.
// Use ap_uint<W> if very large indices (> 2^31) are possible.
typedef int LocationType;

// Stream depth (adjust based on pipeline requirements and buffering needs)
// A small depth is often sufficient for tightly coupled loops.
constexpr int STREAM_DEPTH = 2;

//--------------------------------------------------------------------------
// Type Aliases for Streams
//--------------------------------------------------------------------------

// Input stream for PSS correlation magnitude squared values
using XcorrStream = hls::stream<DataType, STREAM_DEPTH>;

// Input stream for threshold values
using ThresholdStream = hls::stream<DataType, STREAM_DEPTH>;

// Output stream for detected peak locations (indices)
using LocationStream = hls::stream<LocationType, STREAM_DEPTH>;


//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------

/**
 * @brief Implements the peak picker algorithm for SSB detection.
 *
 * Finds peaks in the PSS correlation magnitude squared (`xcorrStream`)
 * where the value exceeds a corresponding `thresholdStream` value and is
 * the local maximum within a sliding window of `WINDOW_LENGTH`.
 *
 * @param xcorrStream     Input stream of PSS correlation magnitude squared values (fixed-point).
 * @param thresholdStream Input stream of threshold values (fixed-point), synchronized with xcorrStream.
 * @param numSamples      Total number of samples to process from the input streams.
 * @param locationStream  Output stream where detected peak locations (indices) are written.
 */
void peakPicker(
    XcorrStream&    xcorrStream,
    ThresholdStream& thresholdStream,
    int             numSamples,
    LocationStream& locationStream
);

#endif // PEAK_PICKER_HPP