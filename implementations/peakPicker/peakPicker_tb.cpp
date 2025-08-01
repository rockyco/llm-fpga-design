/* AUTO-EDITED BY DEBUG ASSISTANT */
#include "peakPicker.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>    // For std::abs
#include <limits>   // For numeric_limits
#include <iomanip>  // For std::setprecision

// Define input/output file names
const std::string XCORR_INPUT_FILE = "pssCorrMagSq_3_in.txt"; // Matches MATLAB TB
const std::string THRESHOLD_INPUT_FILE = "threshold_in.txt"; // Matches MATLAB TB
const std::string REF_OUTPUT_FILE = "locations_3_ref.txt";   // Matches MATLAB TB
// const std::string REF_OUTPUT_FILE = "peakLocs_out.txt"; // Alternative reference file name

// Function to read data from a file into a vector of doubles
bool readDataFile(const std::string& filename, std::vector<double>& data) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    double value;
    while (infile >> value) {
        data.push_back(value);
    }
    // Check for read errors (e.g., non-numeric data) after the loop
    if (infile.bad()) {
        std::cerr << "Error: Failed reading data from file: " << filename << std::endl;
        infile.close();
        return false;
    }
    // Check if EOF was reached OR if a formatting error stopped the loop early
    // infile.fail() is true if >> failed (e.g., bad format), but not for EOF
    // infile.eof() is true if >> tried to read past EOF
    if (!infile.eof() && infile.fail()) {
         std::cerr << "Warning: Input format error suspected in file: " << filename << std::endl;
         // Continue, but be aware data might be incomplete
    }
    infile.close();
    if (data.empty() && !infile.eof()) { // Check if file was opened but no data read
       std::cerr << "Warning: No data read from file (or file empty): " << filename << std::endl;
    }
    std::cout << "Read " << data.size() << " values from " << filename << std::endl;
    return true;
}

// Function to read integer data (locations) from a file
bool readIntDataFile(const std::string& filename, std::vector<int>& data) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    int value;
    while (infile >> value) {
        data.push_back(value);
    }
     // Check for read errors (e.g., non-numeric data) after the loop
    if (infile.bad()) {
        std::cerr << "Error: Failed reading data from file: " << filename << std::endl;
        infile.close();
        return false;
    }
    // Check if EOF was reached OR if a formatting error stopped the loop early
    if (!infile.eof() && infile.fail()) {
         std::cerr << "Warning: Input format error suspected in file: " << filename << std::endl;
         // Continue, but be aware data might be incomplete
    }
    infile.close();
     if (data.empty() && !infile.eof()) { // Check if file was opened but no data read
       std::cerr << "Warning: No data read from file (or file empty): " << filename << std::endl;
    }
    std::cout << "Read " << data.size() << " values from " << filename << std::endl;
    return true;
}


int main() {
    std::cout << "--- Starting Peak Picker Testbench ---" << std::endl;

    // --- Data Loading ---
    std::vector<double> xcorrVec, thresholdVec;
    std::vector<int> refLocsVec; // Use int for reference locations from file

    std::cout << "Loading input data..." << std::endl;
    if (!readDataFile(XCORR_INPUT_FILE, xcorrVec)) return 1;
    if (!readDataFile(THRESHOLD_INPUT_FILE, thresholdVec)) return 1;

    std::cout << "Loading reference output data..." << std::endl;
    // Assuming reference file contains 1-based indices from MATLAB
    if (!readIntDataFile(REF_OUTPUT_FILE, refLocsVec)) return 1;

    // Basic input validation
    if (xcorrVec.size() != thresholdVec.size()) {
        std::cerr << "Error: Input xcorr size (" << xcorrVec.size()
                  << ") does not match threshold size (" << thresholdVec.size() << ")" << std::endl;
        return 1;
    }
    if (xcorrVec.empty()) {
         std::cerr << "Error: Input data vectors are empty (or failed to load)." << std::endl;
         return 1;
    }

    int numSamples = xcorrVec.size();
    std::cout << "Number of samples to process: " << numSamples << std::endl;

    // --- Stream Preparation ---
    hls::stream<Data_t> xcorrStream("xcorrStream");
    hls::stream<Data_t> thresholdStream("thresholdStream");
    hls::stream<Index_t> locationStream("locationStream");

    std::cout << "Populating input streams..." << std::endl;
    for (int i = 0; i < numSamples; ++i) {
        // Convert double to fixed-point Data_t
        // Add checks here if concerned about out-of-range conversions, though
        // ap_fixed usually handles this via saturation or wrapping based on config.
        xcorrStream.write(static_cast<Data_t>(xcorrVec[i]));
        thresholdStream.write(static_cast<Data_t>(thresholdVec[i]));
    }
    std::cout << "Input streams populated." << std::endl;

    // --- Call the DUT (Device Under Test) ---
    std::cout << "Calling HLS peakPicker function..." << std::endl;
    peakPicker(xcorrStream, thresholdStream, locationStream, numSamples);
    std::cout << "HLS peakPicker function finished." << std::endl;

    // --- Collect Results ---
    std::vector<Index_t> actualLocsVec; // Store results from DUT (0-based)
    std::cout << "Collecting results from output stream..." << std::endl;
    while (!locationStream.empty()) {
        actualLocsVec.push_back(locationStream.read());
    }
     std::cout << "Collected " << actualLocsVec.size() << " peak locations." << std::endl;

    // --- Verification ---
    std::cout << "Comparing HLS results with reference..." << std::endl;
    bool match = true;
    int errorCount = 0;

    // Compare number of peaks found
    if (actualLocsVec.size() != refLocsVec.size()) {
        std::cerr << "Error: Mismatch in number of detected peaks!" << std::endl;
        std::cerr << "  Expected: " << refLocsVec.size() << std::endl;
        std::cerr << "  Actual:   " << actualLocsVec.size() << std::endl;
        match = false;
        // Don't stop here, try comparing the elements we do have if sizes are different
        // errorCount will increase significantly anyway.
    } else {
        std::cout << "Number of peaks matches reference (" << refLocsVec.size() << ")." << std::endl;
    }

    // Compare actual peak locations element by element
    size_t comparisonLimit = std::min(actualLocsVec.size(), refLocsVec.size());
    for (size_t i = 0; i < comparisonLimit; ++i) {
        // Convert DUT output (ap_uint<INDEX_W>) to int for comparison
        int actualLoc = static_cast<int>(actualLocsVec[i]); // DUT output (0-based)
        int refLoc = refLocsVec[i];                         // Reference file value (assumed 1-based)

        // --- MODIFIED COMPARISON ---
        // Adjust the 1-based reference index to 0-based for comparison
        int expectedLoc_0based = refLoc - 1;

        if (actualLoc != expectedLoc_0based) {
            if (errorCount < 20) { // Print more mismatches if they occur
               std::cerr << "Mismatch at output index " << i << ":" << std::endl;
               std::cerr << "  Expected (0-based): " << expectedLoc_0based << " (from ref file value " << refLoc << ")" << std::endl;
               std::cerr << "  Actual (0-based):   " << actualLoc << std::endl;
            } else if (errorCount == 20) {
               std::cerr << "Further mismatches suppressed..." << std::endl;
            }
            match = false;
            errorCount++;
        }
        // --- END MODIFIED COMPARISON ---
    }

     // Report if sizes mismatched even if no element mismatches were found within comparisonLimit
     if (actualLocsVec.size() != refLocsVec.size()) {
         match = false; // Ensure test fails if sizes differ
         if (errorCount == 0) { // Only print this if no element mismatches were logged
             std::cerr << "Mismatch due to differing number of peaks." << std::endl;
         }
     }

     if (errorCount > 0) {
         std::cerr << "Total mismatches found: " << errorCount << std::endl;
     }


    // --- Report Results ---
    if (match) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "--- Test PASSED ---" << std::endl;
        std::cout << "HLS implementation output matches the reference output (assuming 1-based reference indices)." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        return 0; // Success
    } else {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "--- Test FAILED ---" << std::endl;
        std::cout << "HLS implementation output does NOT match the reference output." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        return 1; // Failure
    }
}