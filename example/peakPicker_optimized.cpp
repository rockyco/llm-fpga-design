#include "peakPicker.hpp"

#ifdef __SYNTHESIS__
#include <hls_stream.h>
#endif

// Optimized streaming-based implementation for lowest latency
void peakPicker(
    hls::stream<data_t>& xcorr_stream,
    hls::stream<data_t>& threshold_stream,
    index_t input_length,
    hls::stream<index_t>& locations_stream,
    index_t* num_peaks
) {
#ifdef __SYNTHESIS__
    // Optimized interface pragmas for streaming
    #pragma HLS INTERFACE axis port=xcorr_stream
    #pragma HLS INTERFACE axis port=threshold_stream
    #pragma HLS INTERFACE axis port=locations_stream
    #pragma HLS INTERFACE s_axilite port=input_length bundle=control
    #pragma HLS INTERFACE m_axi port=num_peaks offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Dataflow optimization for task-level pipelining
    #pragma HLS DATAFLOW
#endif
    
    // Optimized circular buffer implementation using shift registers
    data_t xcorr_window[WINDOW_LENGTH];
    data_t threshold_window[WINDOW_LENGTH];
    
#ifdef __SYNTHESIS__
    #pragma HLS ARRAY_PARTITION variable=xcorr_window complete dim=1
    #pragma HLS ARRAY_PARTITION variable=threshold_window complete dim=1
#endif
    
    // Initialize windows
    init_window: for (int i = 0; i < WINDOW_LENGTH; i++) {
#ifdef __SYNTHESIS__
        #pragma HLS UNROLL
#endif
        xcorr_window[i] = 0;
        threshold_window[i] = 0;
    }
    
    index_t peak_count = 0;
    
    // Main optimized processing loop with streaming
    main_processing: for (index_t idx = 0; idx < input_length; idx++) {
#ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=6001 max=6001 avg=6001
#endif
        
        // Read from streams (single cycle operation)
        data_t xcorr_sample = xcorr_stream.read();
        data_t threshold_sample = threshold_stream.read();
        
        // Optimized shift register using unrolled operations
#ifdef __SYNTHESIS__
        #pragma HLS UNROLL
#endif
        shift_registers: for (int i = WINDOW_LENGTH - 1; i > 0; i--) {
            xcorr_window[i] = xcorr_window[i-1];
            threshold_window[i] = threshold_window[i-1];
        }
        xcorr_window[0] = xcorr_sample;
        threshold_window[0] = threshold_sample;
        
        // Peak detection logic (only after window is filled)
        if (idx >= WINDOW_LENGTH - 1) {
            // Get middle sample
            data_t mid_xcorr = xcorr_window[MIDDLE_LOCATION];
            data_t mid_threshold = threshold_window[MIDDLE_LOCATION];
            
            // Threshold check
            bool above_threshold = (mid_xcorr > mid_threshold);
            
            // Parallel peak comparison using unrolled loop
            bool is_local_max = true;
#ifdef __SYNTHESIS__
            #pragma HLS UNROLL
#endif
            peak_comparison: for (int i = 0; i < WINDOW_LENGTH; i++) {
                if (i != MIDDLE_LOCATION && xcorr_window[i] >= mid_xcorr) {
                    is_local_max = false;
                }
            }
            
            // Output peak location if detected
            if (is_local_max && above_threshold && peak_count < MAX_PEAKS) {
                index_t peak_location = idx - MIDDLE_LOCATION + 1; // MATLAB 1-indexed
                locations_stream.write(peak_location);
                peak_count++;
            }
        }
    }
    
    *num_peaks = peak_count;
}

// Wrapper function for backward compatibility with array interface
void peakPicker_wrapper(
    data_t xcorr[MAX_INPUT_SIZE],
    data_t threshold[MAX_INPUT_SIZE], 
    index_t input_length,
    index_t locations[MAX_PEAKS],
    index_t* num_peaks
) {
#ifdef __SYNTHESIS__
    // Interface pragmas for wrapper
    #pragma HLS INTERFACE m_axi port=xcorr offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=threshold offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=locations offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=num_peaks offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=input_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    #pragma HLS DATAFLOW
#endif
    
    // Create streams
    static hls::stream<data_t> xcorr_stream("xcorr_stream");
    static hls::stream<data_t> threshold_stream("threshold_stream");
    static hls::stream<index_t> locations_stream("locations_stream");
    
#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=xcorr_stream depth=2
    #pragma HLS STREAM variable=threshold_stream depth=2
    #pragma HLS STREAM variable=locations_stream depth=100
#endif
    
    // Feed input streams
    input_feeder: for (index_t i = 0; i < input_length; i++) {
#ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
#endif
        xcorr_stream.write(xcorr[i]);
        threshold_stream.write(threshold[i]);
    }
    
    // Call optimized core function
    index_t temp_num_peaks;
    peakPicker(xcorr_stream, threshold_stream, input_length, locations_stream, &temp_num_peaks);
    
    // Read output stream
    output_collector: for (index_t i = 0; i < temp_num_peaks && i < MAX_PEAKS; i++) {
#ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
#endif
        locations[i] = locations_stream.read();
    }
    
    *num_peaks = temp_num_peaks;
}