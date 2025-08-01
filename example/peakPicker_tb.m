% This is the testbench for the peakPicker function.
% It reads the input signal from a file, calls the peakPicker function,
% and writes the output to a file.
%
% Author: Jie Lei
% Date: 03/30/2025
%
% Read the input cross correlation from a file.
xcorr = readmatrix('pssCorrMagSq_3_in.txt','Delimiter', 'tab');
% Read the threshold from a file.
threshold = readmatrix('threshold_in.txt','Delimiter', 'tab');

% Call the peakPicker function.
[peakLocs] = peakPicker(xcorr, threshold);
% Write the output to a file.
writematrix(peakLocs, 'peakLocs_out.txt','Delimiter', 'tab');
% Read the reference output from a file.
refLocs = readmatrix('locations_3_ref.txt','Delimiter', 'tab');
% Compare the output with the reference output.
if isequal(peakLocs, refLocs)
    disp('Test passed: The output matches the reference output.');
else
    disp('Test failed: The output does not match the reference output.');
end