#ifndef PEAKPICKER_HPP
#define PEAKPICKER_HPP

#ifdef __SYNTHESIS__
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>

// Optimized data type definitions for lowest latency
typedef ap_fixed<24, 6> data_t;      // Reduced precision: 24-bit, 6 integer bits
typedef ap_uint<16> index_t;         // 16-bit unsigned integer for indices
typedef ap_uint<8> count_t;          // 8-bit counter for small counts
#else
// Testbench mode - use standard C++ types
typedef double data_t;               // Use double for testbench compatibility
typedef unsigned short index_t;      // 16-bit unsigned integer
typedef unsigned char count_t;       // 8-bit counter
#endif

// Algorithm parameters
constexpr int WINDOW_LENGTH = 11;
constexpr int MIDDLE_LOCATION = WINDOW_LENGTH / 2;  // 5
constexpr int MAX_INPUT_SIZE = 6001;
constexpr int MAX_PEAKS = 100;  // Conservative estimate for maximum peaks

#ifdef __SYNTHESIS__
// Optimized streaming interface function declaration
void peakPicker(
    hls::stream<data_t>& xcorr_stream,
    hls::stream<data_t>& threshold_stream,
    index_t input_length,
    hls::stream<index_t>& locations_stream,
    index_t* num_peaks
);
#endif

// Array-based interface for testbench compatibility and synthesis
void peakPicker_wrapper(
    data_t xcorr[MAX_INPUT_SIZE],
    data_t threshold[MAX_INPUT_SIZE], 
    index_t input_length,
    index_t locations[MAX_PEAKS],
    index_t* num_peaks
);

// Main function declaration
void peakPicker(
    data_t xcorr[MAX_INPUT_SIZE],
    data_t threshold[MAX_INPUT_SIZE], 
    index_t input_length,
    index_t locations[MAX_PEAKS],
    index_t* num_peaks
);

#endif // PEAKPICKER_HPP