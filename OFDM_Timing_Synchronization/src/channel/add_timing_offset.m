function rxWindow = add_timing_offset(rxSignal, idealStart, tau, Nfft)
% ADD_TIMING_OFFSET
% Extract an FFT window with a specified timing offset.
% Inputs:
%   rxSignal - Received signal with sufficient buffer
%   idealStart - Ideal FFT window starting index
%   tau - Timing offset in samples
%   Nfft - FFT size
%Output:
%rxWindow - Extracted FFT window

%Calculate the actual FFT window start
actualStart = idealStart + tau;

%Calculate the FFT window end
actualEnd = actualStart + Nfft - 1;

%Check whether the window is valid
if actualStart < 1 || actualEnd > numel(rxSignal)
    error('Timing offset exceeds the available signal buffer.');
end

%Extract the FFT window
rxWindow = rxSignal(actualStart:actualEnd);

end