function [locations]= peakPicker(xcorr,threshold,window_length)
% Copyright 2021-2023 The MathWorks, Inc.

    locations=[];
    middle_location=floor(window_length/2);
    
    for index=1:length(xcorr)-window_length+1
        current_window=xcorr(index:index+window_length-1,:); % sliding window
        candidate_location=index+middle_location;
        if any(xcorr(candidate_location,:)>=threshold(candidate_location)) % check if middle value clears threshold
            seqNumber = find(xcorr(candidate_location,:)>=threshold(candidate_location));
            if sum(xcorr(candidate_location,seqNumber)>=current_window(:,seqNumber))==window_length % check if maximum value
                locations=[locations;candidate_location]; %#ok
            end
        end
    end
end