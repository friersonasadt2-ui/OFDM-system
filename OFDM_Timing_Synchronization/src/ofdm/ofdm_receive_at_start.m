function [bitsHat, isValid, equalizedSymbols] = ...
    ofdm_receive_at_start(rx,fftStart,cfg,H)
% OFDM_RECEIVE_AT_START
%
% Demodulate one QPSK-OFDM data symbol
% using a specified useful FFT-window start.
%
% rx       : received sample buffer
% fftStart : 1-based useful FFT start
% cfg      : OFDM configuration
% H        : known frequency-domain channel response
%
% No true timing reference or true timing error
% is used internally.

N = cfg.Nfft;

rx = rx(:);
H = H(:);

bitsHat = [];
equalizedSymbols = [];
isValid = false;

%% 1. Validate channel response

assert(numel(H)==N, ...
    'Channel response length mismatch.');

assert(all(isfinite(H)), ...
    'Invalid channel frequency response.');

%% 2. Validate FFT start

if ~isscalar(fftStart) || ...
        ~isfinite(fftStart) || ...
        fftStart~=round(fftStart)

    return;

end

fftEnd = fftStart+N-1;

if fftStart<1 || fftEnd>numel(rx)
    return;
end

%% 3. Extract useful OFDM samples

rxWindow = rx(fftStart:fftEnd);

if any(~isfinite(rxWindow))
    return;
end

%% 4. FFT

Y = fft(rxWindow,N);

%% 5. Zero-forcing channel equalization

if any(abs(H)<1e-12)
    return;
end

equalizedSymbols = Y./H;

if any(~isfinite(equalizedSymbols))
    return;
end

%% 6. QPSK demodulation

bitsHat = qpsk_demod(equalizedSymbols);

bitsHat = bitsHat(:);

assert(numel(bitsHat)==N*cfg.bitsPerSymbol, ...
    'Decoded bit length mismatch.');

assert(all(bitsHat==0 | bitsHat==1), ...
    'Invalid decoded bits.');

isValid = true;

end