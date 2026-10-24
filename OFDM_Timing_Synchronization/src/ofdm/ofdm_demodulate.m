function symbols = ofdm_demodulate(rxSignal, cfg)
%OFDM_DEMODULATE Remove cyclic prefix and perform OFDM demodulation.
%symbols = ofdm_demodulate(rxSignal, cfg
%% Length of one OFDM symbol including CP
symbolLength = cfg.Nfft + cfg.Ncp;

if mod(length(rxSignal), symbolLength) ~= 0
    error('Received signal length is not a multiple of OFDM symbol length.');
end
%% Convert serial signal back into OFDM symbols
rxMatrix = reshape(rxSignal, symbolLength, []);

%% Remove cyclic prefix
timeData = rxMatrix(cfg.Ncp+1:end, :);

%% FFT: time domain -> frequency domain
freqData = fft(timeData, cfg.Nfft, 1);

%% Convert back into serial QPSK symbols
symbols = freqData(:);
end