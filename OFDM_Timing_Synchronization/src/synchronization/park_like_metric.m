function [P, M, centerIndex] = park_like_metric(rx, Nfft)
% Normalized mirror-correlation timing metric.
% Modified Park-like diagnostic, NOT standard Park (2003).
%
% centerIndex is the first sample of the right half.
% Estimated useful start = centerIndex - Nfft/2.

rx = rx(:);

assert(mod(Nfft,2)==0, 'Nfft must be even.');
assert(numel(rx)>=Nfft, 'Received signal is too short.');

L = Nfft/2;

centerIndex = (L+1:numel(rx)-L+1).';

P = zeros(size(centerIndex));
M = zeros(size(centerIndex));

for i = 1:numel(centerIndex)

    s = centerIndex(i);

    left = rx(s-L:s-1);
    right = rx(s:s+L-1);

    P(i) = sum(left .* flipud(right));

    EL = sum(abs(left).^2);
    ER = sum(abs(right).^2);

    if EL > 1e-12 && ER > 1e-12
        M(i) = abs(P(i))^2/(EL*ER);
    else
        M(i) = 0;
    end
end
end