function bits = qpsk_demod(symbols)
% QPSK_DEMOD Demodulate QPSK symbols into binary bits.
% bits = qpsk_demod(symbols)

%% QPSK demodulation
symbolIndices = pskdemod(symbols, 4, pi/4);
%% Convert integers back to bits
bitPairs = de2bi(symbolIndices, 2, 'left-msb');
%% Convert matrix back to column vector
bits = reshape(bitPairs.', [], 1);
end