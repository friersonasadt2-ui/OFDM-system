function [P, R, M, centerIndex] = park_metric_original(rx, Nfft)

rx = rx(:);

assert(mod(Nfft, 2) == 0, 'Nfft must be even.');

L = Nfft / 2;
numSamples = numel(rx);

assert(numSamples >= Nfft, ...
    'Received signal must contain at least Nfft samples.');

centerIndex = (L+1:numSamples-L+1).';

P = zeros(size(centerIndex));
R = zeros(size(centerIndex));
M = zeros(size(centerIndex));

for i = 1:numel(centerIndex)

    s = centerIndex(i);

    left = rx(s-L:s-1);
    right = rx(s:s+L-1);

    P(i) = sum(left .* flipud(right));
    R(i) = sum(abs(right).^2);

    if R(i) > 1e-12
        M(i) = abs(P(i))^2 / R(i)^2;
    end

end

end