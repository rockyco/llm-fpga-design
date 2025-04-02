#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include "peakPicker.hpp" // Include the header for the HLS function

// Helper function to read matrix data (float) from a file
// Determines dimensions dynamically.
bool readMatrixFromFile(const std::string& filename, std::vector<std::vector<float>>& matrix, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << filename << std::endl;
        return false;
    }

    matrix.clear();
    std::string line;
    rows = 0;
    cols = 0;

    while (std::getline(file, line)) {
        std::vector<float> rowVec;
        std::stringstream ss(line);
        float value;
        int currentCol = 0;

        // Use tab as delimiter, matching MATLAB writematrix default
        std::string cell;
        while (std::getline(ss, cell, '\t')) {
             try {
                rowVec.push_back(std::stof(cell));
                currentCol++;
            } catch (const std::invalid_argument& e) {
                std::cerr << "ERROR: Invalid number format in " << filename << " at row " << rows + 1 << ", content: '" << cell << "'" << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "ERROR: Number out of range in " << filename << " at row " << rows + 1 << ", content: '" << cell << "'" << std::endl;
                return false;
            }
        }


        if (rows == 0) {
            cols = currentCol; // Set column count based on the first row
        } else if (currentCol != cols) {
            std::cerr << "ERROR: Inconsistent number of columns in " << filename << " at row " << rows + 1 << ". Expected " << cols << ", found " << currentCol << "." << std::endl;
            return false;
        }

        if (currentCol > 0) { // Only add non-empty rows
             matrix.push_back(rowVec);
             rows++;
        }
    }

    file.close();
    if (rows == 0 || cols == 0) {
         std::cerr << "ERROR: No data read or empty file: " << filename << std::endl;
         return false;
    }
    std::cout << "INFO: Read " << rows << " rows and " << cols << " columns from " << filename << std::endl;
    return true;
}

// Helper function to read vector data (int) from a file
bool readVectorFromFile(const std::string& filename, std::vector<int>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << filename << std::endl;
        return false;
    }

    vec.clear();
    std::string line;
    int value;
    while (std::getline(file, line)) {
         // Handle potential empty lines or lines with just whitespace
        std::stringstream ss(line);
        if (ss >> value) {
             vec.push_back(value);
        } else if (!line.empty() && line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
            // Line is not empty and not just whitespace, but couldn't parse an int
            std::cerr << "ERROR: Invalid integer format in " << filename << " line: '" << line << "'" << std::endl;
            file.close();
            return false;
        }
    }

    file.close();
    std::cout << "INFO: Read " << vec.size() << " elements from " << filename << std::endl;
    return true;
}

// Helper function to write vector data (int) to a file
bool writeVectorToFile(const std::string& filename, const std::vector<Index_t>& vec) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        file << vec[i];
        if (i < vec.size() - 1) {
            file << "\t"; // Use tab delimiter, matching MATLAB default
        } else {
             file << std::endl; // Newline after the last element
        }
    }
     // Ensure a newline even if the vector is empty, matching writematrix behavior
    if (vec.empty()) {
        file << std::endl;
    }


    file.close();
    std::cout << "INFO: Wrote " << vec.size() << " elements to " << filename << std::endl;
    return true;
}


int main() {
    // --- Configuration ---
    const std::string XCORR_IN_FILE = "pssCorrMagSq_3_in.txt";
    const std::string THRESH_IN_FILE = "threshold_in.txt";
    const std::string LOCS_REF_FILE = "locations_3_ref.txt";
    const std::string LOCS_OUT_FILE = "peakLocs_3_out.txt"; // Output file for C++ results

    std::cout << "--- Peak Picker HLS Testbench ---" << std::endl;
    std::cout << "INFO: Using NUM_SEQUENCES = " << NUM_SEQUENCES << std::endl;
    std::cout << "INFO: Using WINDOW_LENGTH = " << WINDOW_LENGTH << std::endl;
    std::cout << "INFO: Using Fixed-Point Type: ap_fixed<" << DATA_W << ", " << DATA_I << ">" << std::endl;

    // --- Read Input Data ---
    std::vector<std::vector<float>> xcorrFloat;
    std::vector<std::vector<float>> thresholdFloat;
    int dataLength = 0;
    int numXcorrCols = 0;
    int numThreshCols = 0;
    int threshRows = 0;

    std::cout << "INFO: Reading input files..." << std::endl;
    if (!readMatrixFromFile(XCORR_IN_FILE, xcorrFloat, dataLength, numXcorrCols)) return 1;
    if (!readMatrixFromFile(THRESH_IN_FILE, thresholdFloat, threshRows, numThreshCols)) return 1;

    // --- Validate Input Dimensions ---
    if (numXcorrCols != NUM_SEQUENCES) {
        std::cerr << "ERROR: Mismatch between NUM_SEQUENCES (" << NUM_SEQUENCES
                  << ") and columns read from " << XCORR_IN_FILE << " (" << numXcorrCols << ")" << std::endl;
        return 1;
    }
    if (numThreshCols != NUM_SEQUENCES) {
        std::cerr << "ERROR: Mismatch between NUM_SEQUENCES (" << NUM_SEQUENCES
                  << ") and columns read from " << THRESH_IN_FILE << " (" << numThreshCols << ")" << std::endl;
        return 1;
    }
    if (threshRows != dataLength) {
        std::cerr << "ERROR: Row count mismatch between " << XCORR_IN_FILE << " (" << dataLength
                  << ") and " << THRESH_IN_FILE << " (" << threshRows << ")" << std::endl;
        return 1;
    }
     if (dataLength < WINDOW_LENGTH) {
        std::cerr << "ERROR: Data length (" << dataLength << ") is less than WINDOW_LENGTH (" << WINDOW_LENGTH << ")." << std::endl;
        return 1;
    }


    // --- Read Reference Data ---
    std::vector<int> refLocationsMatlab; // Stores 1-based indices from MATLAB file
    std::cout << "INFO: Reading reference file..." << std::endl;
    if (!readVectorFromFile(LOCS_REF_FILE, refLocationsMatlab)) return 1;

    // Convert reference locations to 0-based for comparison with C++ output
    std::vector<Index_t> refLocationsZeroBased;
    refLocationsZeroBased.reserve(refLocationsMatlab.size());
    for (int loc : refLocationsMatlab) {
        if (loc < 1) {
             std::cerr << "ERROR: Invalid 1-based index found in reference file " << LOCS_REF_FILE << ": " << loc << std::endl;
             return 1;
        }
        refLocationsZeroBased.push_back(loc - 1);
    }

    // --- Prepare HLS Streams ---
    hls::stream<InputSample_t> xcorrStream("xcorrStream");
    hls::stream<InputSample_t> thresholdStream("thresholdStream");
    hls::stream<Index_t> locationsStream("locationsStream");

    std::cout << "INFO: Populating input HLS streams..." << std::endl;
    for (int i = 0; i < dataLength; ++i) {
        InputSample_t xcorrSampleVec;
        InputSample_t thresholdSampleVec;
        for (int s = 0; s < NUM_SEQUENCES; ++s) {
            // Convert float to fixed-point
            xcorrSampleVec[s] = Data_t(xcorrFloat[i][s]);
            thresholdSampleVec[s] = Data_t(thresholdFloat[i][s]);
        }
        xcorrStream.write(xcorrSampleVec);
        thresholdStream.write(thresholdSampleVec);
    }
    std::cout << "INFO: Input streams populated with " << dataLength << " samples." << std::endl;

    // --- Execute DUT (Device Under Test) ---
    std::cout << "INFO: Calling HLS peakPicker function..." << std::endl;
    peakPicker(xcorrStream, thresholdStream, locationsStream, dataLength);
    std::cout << "INFO: HLS function execution finished." << std::endl;

    // --- Collect Output ---
    std::vector<Index_t> dutLocations;
    std::cout << "INFO: Reading output HLS stream..." << std::endl;
    while (!locationsStream.empty()) {
        dutLocations.push_back(locationsStream.read());
    }
    std::cout << "INFO: Collected " << dutLocations.size() << " peak locations from DUT." << std::endl;

    // --- Write DUT Output to File ---
    std::cout << "INFO: Writing DUT output to " << LOCS_OUT_FILE << "..." << std::endl;
    if (!writeVectorToFile(LOCS_OUT_FILE, dutLocations)) return 1;

    // --- Verification ---
    std::cout << "INFO: Verifying DUT output against reference..." << std::endl;
    bool match = true;
    if (dutLocations.size() != refLocationsZeroBased.size()) {
        std::cerr << "ERROR: Size mismatch! DUT produced " << dutLocations.size()
                  << " locations, Reference has " << refLocationsZeroBased.size() << "." << std::endl;
        match = false;
    } else {
        for (size_t i = 0; i < dutLocations.size(); ++i) {
            if (dutLocations[i] != refLocationsZeroBased[i]) {
                std::cerr << "ERROR: Mismatch at index " << i << "! DUT: " << dutLocations[i]
                          << ", Reference (0-based): " << refLocationsZeroBased[i] << std::endl;
                match = false;
                // Optional: break after first mismatch or report all
                // break;
            }
        }
    }

    // --- Report Result ---
    if (match) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "--- TEST PASSED ---" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        return 0; // Success
    } else {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "--- TEST FAILED ---" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        // Print detailed comparison if sizes differ significantly or upon request
        if (dutLocations.size() != refLocationsZeroBased.size()) {
             std::cout << "DUT Locations (" << dutLocations.size() << "): ";
             for(const auto& loc : dutLocations) std::cout << loc << " ";
             std::cout << std::endl;
             std::cout << "Ref Locations (0-based, " << refLocationsZeroBased.size() << "): ";
             for(const auto& loc : refLocationsZeroBased) std::cout << loc << " ";
             std::cout << std::endl;
        }
        return 1; // Failure
    }
}
