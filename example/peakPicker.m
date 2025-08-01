% Modified from the original peakPicker.m file
% to use a hardware-friendly implementation of peak finder
% This function is used to find the locations of peaks in a cross-correlation
% signal. It takes the cross-correlation signal, a threshold, and a window
% length as inputs, and returns the locations of the detected peaks.
% The function uses a sliding window approach to check for local maxima
% within the specified window length. The middle sample of the window is
% compared to the other samples in the window, and if it is greater than
% the threshold, it is considered a peak. The function also ensures that
% the detected peaks are within the specified window length.
% The function is designed to be hardware-friendly, meaning it is optimized
% for implementation on hardware platforms such as FPGAs or ASICs. It uses
% a simple and efficient algorithm to find the peaks, avoiding complex
% operations that may not be suitable for hardware implementation.
% The function is written in MATLAB and can be used in various applications
% such as signal processing, communications, and data analysis.
%
% The function takes the following inputs:
% - xcorr: The cross-correlation signal, which is a matrix of size
%   (num_samples, num_sequences). Each column represents a different
%   sequence.
% - threshold: The threshold value for peak detection, which is a vector
%   of size (num_samples, 1). The threshold is used to determine if a
%   sample is considered a peak.
% - window_length: The length of the sliding window used for peak
%   detection. It is a scalar value that specifies the number of samples
%   to consider in the window.
%
% The function returns the following output:
% - locations: A vector containing the indices of the detected peaks in
%   the cross-correlation signal. The indices are relative to the input
%   signal and indicate the locations of the detected peaks.
%
% Author: Jie Lei
% Date: 03/31/2025
% University of Technology Sydney

function [locations]= peakPicker(xcorr,threshold)
% Copyright 2021-2023 The MathWorks, Inc.

    locations=[];
    window_length = 11; % Length of the sliding window
    middle_location=floor(window_length/2);
    xcorrBuffer = zeros(window_length, 1); % Preallocate buffer for current window
    thresholdBuffer = zeros(window_length, 1); % Preallocate buffer for threshold
    
    for index=1:length(xcorr)-window_length+1
        xcorrBuffer(2:end) = xcorrBuffer(1:end-1); % Shift buffer
        xcorrBuffer(1) = xcorr(index); % Add new sample to buffer
        thresholdBuffer(2:end) = thresholdBuffer(1:end-1); % Shift threshold buffer
        thresholdBuffer(1) = threshold(index); % Add new threshold to buffer
        if (index >= window_length)
            candidate_location = index - middle_location;
            % Hardware friendly implementation of peak finder
            MidSample = xcorrBuffer(middle_location+1,:);
            CompareOut = xcorrBuffer - MidSample; % this is a vector
            % if all values in the result are negative and the middle sample is
            % greater than a threshold, it is a local max
            if all(CompareOut <= 0) && (MidSample > thresholdBuffer(middle_location+1))
                locations = [locations candidate_location]; %#ok
            end
        end
    end
end
