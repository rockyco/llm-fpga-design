#include "peakPicker.hpp"

#ifdef __SYNTHESIS__
#include <hls_stream.h>
#endif

// Ultra-optimized implementation targeting II=1 
void peakPicker_wrapper(
    data_t xcorr[MAX_INPUT_SIZE],
    data_t threshold[MAX_INPUT_SIZE], 
    index_t input_length,
    index_t locations[MAX_PEAKS],
    index_t* num_peaks
) {
#ifdef __SYNTHESIS__
    // Optimized interface pragmas - consolidated AXI bundles
    #pragma HLS INTERFACE m_axi port=xcorr offset=slave bundle=gmem max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=threshold offset=slave bundle=gmem max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=locations offset=slave bundle=gmem_out max_write_burst_length=100
    #pragma HLS INTERFACE m_axi port=num_peaks offset=slave bundle=gmem_out
    #pragma HLS INTERFACE s_axilite port=input_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
#endif
    
    // Optimized circular buffer using explicit shift register pattern
    data_t xcorr_sr[WINDOW_LENGTH];
    data_t threshold_sr[WINDOW_LENGTH];
    
#ifdef __SYNTHESIS__
    // Complete array partitioning for parallel access
    #pragma HLS ARRAY_PARTITION variable=xcorr_sr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=threshold_sr complete dim=1
#endif
    
    // Initialize shift registers
    init_sr: for (int i = 0; i < WINDOW_LENGTH; i++) {
#ifdef __SYNTHESIS__
        #pragma HLS UNROLL
#endif
        xcorr_sr[i] = 0;
        threshold_sr[i] = 0;
    }
    
    index_t peak_count = 0;
    
    // Ultra-optimized main loop - processes all samples including initial window fill
    ultra_main_loop: for (index_t idx = 0; idx < input_length; idx++) {
#ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=6001 max=6001 avg=6001
        // Force dependency analysis to avoid false dependencies
        #pragma HLS DEPENDENCE variable=xcorr_sr inter false
        #pragma HLS DEPENDENCE variable=threshold_sr inter false
#endif
        
        // Read new samples
        data_t new_xcorr = xcorr[idx];
        data_t new_threshold = threshold[idx];
        
        // Explicit shift register implementation (completely unrolled)
        // This avoids memory dependency issues
        // Manual shift register unrolling for maximum performance
        xcorr_sr[10] = xcorr_sr[9];
        xcorr_sr[9] = xcorr_sr[8];
        xcorr_sr[8] = xcorr_sr[7];
        xcorr_sr[7] = xcorr_sr[6];
        xcorr_sr[6] = xcorr_sr[5];
        xcorr_sr[5] = xcorr_sr[4];
        xcorr_sr[4] = xcorr_sr[3];
        xcorr_sr[3] = xcorr_sr[2];
        xcorr_sr[2] = xcorr_sr[1];
        xcorr_sr[1] = xcorr_sr[0];
        xcorr_sr[0] = new_xcorr;
        
        threshold_sr[10] = threshold_sr[9];
        threshold_sr[9] = threshold_sr[8];
        threshold_sr[8] = threshold_sr[7];
        threshold_sr[7] = threshold_sr[6];
        threshold_sr[6] = threshold_sr[5];
        threshold_sr[5] = threshold_sr[4];
        threshold_sr[4] = threshold_sr[3];
        threshold_sr[3] = threshold_sr[2];
        threshold_sr[2] = threshold_sr[1];
        threshold_sr[1] = threshold_sr[0];
        threshold_sr[0] = new_threshold;
        
        // Peak detection (starts after window is filled)
        if (idx >= WINDOW_LENGTH - 1) {
            // Get middle sample (index 5 for window of 11)
            data_t mid_xcorr = xcorr_sr[MIDDLE_LOCATION];
            data_t mid_threshold = threshold_sr[MIDDLE_LOCATION];
            
            // Threshold check
            bool above_threshold = (mid_xcorr > mid_threshold);
            
            // Parallel peak detection - fully unrolled comparison
            bool is_peak = (xcorr_sr[0] <= mid_xcorr) && 
                          (xcorr_sr[1] <= mid_xcorr) && 
                          (xcorr_sr[2] <= mid_xcorr) && 
                          (xcorr_sr[3] <= mid_xcorr) && 
                          (xcorr_sr[4] <= mid_xcorr) && 
                          // Skip middle element (index 5)
                          (xcorr_sr[6] <= mid_xcorr) && 
                          (xcorr_sr[7] <= mid_xcorr) && 
                          (xcorr_sr[8] <= mid_xcorr) && 
                          (xcorr_sr[9] <= mid_xcorr) && 
                          (xcorr_sr[10] <= mid_xcorr);
            
            // Compute peak location
            index_t peak_location = idx - MIDDLE_LOCATION + 1; // MATLAB 1-indexed
            
            // Conditional peak storage
            if (is_peak && above_threshold && peak_count < MAX_PEAKS) {
                locations[peak_count] = peak_location;
                peak_count++;
            }
        }
    }
    
    *num_peaks = peak_count;
}

// Alias for backward compatibility
void peakPicker(
    data_t xcorr[MAX_INPUT_SIZE],
    data_t threshold[MAX_INPUT_SIZE], 
    index_t input_length,
    index_t locations[MAX_PEAKS],
    index_t* num_peaks
) {
    peakPicker_wrapper(xcorr, threshold, input_length, locations, num_peaks);
}