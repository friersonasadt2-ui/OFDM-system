function txSignal = ofdm_modulate(symbols, cfg)
%OFDM_MODULATE Perform OFDM modulation and add cyclic prefix.
%txSignal = ofdm_modulate(symbols, cfg)
%Input:
%symbols - QPSK symbols
%cfg - Project configuration structure
%Output:
%txSignal - Time-domain OFDM signal with cyclic prefix
%% Check input size
if mod(length(symbols), cfg.Nfft) ~= 0
    error('Number of QPSK symbols must be a multiple of Nfft.');
end
%% Arrange symbols into OFDM symbols
freqData = reshape(symbols, cfg.Nfft, []);
%% IFFT: frequency domain -> time domain
timeData = ifft(freqData, cfg.Nfft, 1);
%% Add cyclic prefix
cyclicPrefix = timeData(end-cfg.Ncp+1:end, :);
timeDataWithCP = [cyclicPrefix; timeData];
%% Convert all OFDM symbols into one serial signal
txSignal = timeDataWithCP(:);

end