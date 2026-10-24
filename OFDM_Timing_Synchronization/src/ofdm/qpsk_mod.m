function symbols = qpsk_mod(bits)
% QPSK_MOD modulate binary bits using QPSK
% symbols = qpsk_mod(bits)
% Input: bits - Column vector of binary bits
% Output: symbols - Column vector of complex QPSK symbols
%% Check input length
if mod(length(bits), 2) ~=0
    error('The number of input bits must be even for QPSK.');
end
%% Convert every two bits to one integer
bitPairs = reshape(bits, 2, []).';
symbolIndices = bi2de(bitPairs, 'left-msb');
%% QPSK modulation
symbols = pskmod(symbolIndices, 4, pi/4);
end