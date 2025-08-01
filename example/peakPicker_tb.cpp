#include "peakPicker.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

// Function to read data from file
vector<double> readDataFromFile(const string& filename) {
    vector<double> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }
    
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    
    file.close();
    cout << "Read " << data.size() << " values from " << filename << endl;
    return data;
}

// Function to read reference locations
vector<int> readReferenceLocations(const string& filename) {
    vector<int> locations;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return locations;
    }
    
    int value;
    while (file >> value) {
        locations.push_back(value);
    }
    
    file.close();
    cout << "Read " << locations.size() << " reference locations from " << filename << endl;
    return locations;
}

// Function to write results to file
void writeResultsToFile(const string& filename, const vector<int>& locations) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not create file " << filename << endl;
        return;
    }
    
    for (size_t i = 0; i < locations.size(); i++) {
        file << locations[i];
        if (i < locations.size() - 1) {
            file << "\t";
        }
    }
    file << endl;
    
    file.close();
    cout << "Written " << locations.size() << " locations to " << filename << endl;
}

int main() {
    cout << "=== Peak Picker HLS Testbench ===" << endl;
    
    // Read input data
    vector<double> xcorr_data = readDataFromFile("pssCorrMagSq_3_in.txt");
    vector<double> threshold_data = readDataFromFile("threshold_in.txt");
    vector<int> ref_locations = readReferenceLocations("locations_3_ref.txt");
    
    if (xcorr_data.empty() || threshold_data.empty()) {
        cerr << "Error: Failed to read input data files" << endl;
        return -1;
    }
    
    if (xcorr_data.size() != threshold_data.size()) {
        cerr << "Error: Input data size mismatch" << endl;
        return -1;
    }
    
    cout << "Input data size: " << xcorr_data.size() << " samples" << endl;
    
    // Prepare data for HLS function
    static data_t xcorr[MAX_INPUT_SIZE];
    static data_t threshold[MAX_INPUT_SIZE];
    static index_t locations[MAX_PEAKS];
    index_t num_peaks = 0;
    
    // Convert input data to fixed-point
    index_t input_length = min((size_t)MAX_INPUT_SIZE, xcorr_data.size());
    
    for (index_t i = 0; i < input_length; i++) {
        xcorr[i] = (data_t)xcorr_data[i];
        threshold[i] = (data_t)threshold_data[i];
    }
    
    // Initialize output array
    for (int i = 0; i < MAX_PEAKS; i++) {
        locations[i] = 0;
    }
    
    cout << "Calling peakPicker function..." << endl;
    
    // Call the HLS function
    peakPicker(xcorr, threshold, input_length, locations, &num_peaks);
    
    cout << "Peak detection completed. Found " << num_peaks << " peaks." << endl;
    
    // Convert results to vector for easier handling
    vector<int> detected_locations;
    for (index_t i = 0; i < num_peaks; i++) {
        detected_locations.push_back((int)locations[i]);
    }
    
    // Write results to file
    writeResultsToFile("peakLocs_out.txt", detected_locations);
    
    // Compare with reference
    cout << "\n=== Results Comparison ===" << endl;
    cout << "Detected peaks: " << detected_locations.size() << endl;
    cout << "Reference peaks: " << ref_locations.size() << endl;
    
    if (detected_locations.size() != ref_locations.size()) {
        cout << "WARNING: Different number of peaks detected!" << endl;
    }
    
    // Print detected locations
    cout << "\nDetected peak locations: ";
    for (size_t i = 0; i < detected_locations.size(); i++) {
        cout << detected_locations[i];
        if (i < detected_locations.size() - 1) cout << ", ";
    }
    cout << endl;
    
    // Print reference locations
    cout << "Reference peak locations: ";
    for (size_t i = 0; i < ref_locations.size(); i++) {
        cout << ref_locations[i];
        if (i < ref_locations.size() - 1) cout << ", ";
    }
    cout << endl;
    
    // Check if results match
    bool results_match = true;
    if (detected_locations.size() == ref_locations.size()) {
        for (size_t i = 0; i < detected_locations.size(); i++) {
            if (detected_locations[i] != ref_locations[i]) {
                results_match = false;
                break;
            }
        }
    } else {
        results_match = false;
    }
    
    cout << "\n=== Test Result ===" << endl;
    if (results_match) {
        cout << "✓ TEST PASSED: Output matches reference" << endl;
        return 0;
    } else {
        cout << "✗ TEST FAILED: Output does not match reference" << endl;
        
        // Calculate error metrics if sizes match
        if (detected_locations.size() == ref_locations.size() && !detected_locations.empty()) {
            double total_abs_error = 0;
            double max_abs_error = 0;
            
            for (size_t i = 0; i < detected_locations.size(); i++) {
                double abs_error = abs(detected_locations[i] - ref_locations[i]);
                total_abs_error += abs_error;
                max_abs_error = max(max_abs_error, abs_error);
            }
            
            double mean_abs_error = total_abs_error / detected_locations.size();
            
            cout << "Error Analysis:" << endl;
            cout << "  Mean absolute error: " << mean_abs_error << endl;
            cout << "  Maximum absolute error: " << max_abs_error << endl;
        }
        
        return 1;
    }
}