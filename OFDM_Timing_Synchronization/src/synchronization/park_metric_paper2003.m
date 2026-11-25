function [P,R,M,centerIndex] = park_metric_paper2003(rx,Nfft)
% PARK_METRIC_PAPER2003
% Supervisor-confirmed Park timing metric implementation.
% Correlation is performed around the boundary
% between samples d-1 and d.
% P(d) = abs(sum(r(d-k).*r(d+k-1)))
% M(d) = abs(P(d))^2 / max(abs(P)^2)
% R is retained as the right-window energy
% for diagnostic purposes.

rx = rx(:);

assert(mod(Nfft,2)==0, ...
    'Nfft must be even.');

L = Nfft/2;
numSamples = numel(rx);

assert(numSamples >= Nfft, ...
    'Received signal is too short.');

% 1. Valid correlation centers
centerIndex = (L+1:numSamples-L+1).';

P = zeros(size(centerIndex));
R = zeros(size(centerIndex));
M = zeros(size(centerIndex));

k = (1:L).';

% 2. Calculate correlation and energy
for i = 1:numel(centerIndex)

    d = centerIndex(i);

    left = rx(d-k);
    right = rx(d+k-1);

    P(i) = abs(sum(left .* right));

    R(i) = sum(abs(right).^2);

end

% 3. Supervisor-confirmed normalization
correlationPower = abs(P).^2;
maxPower = max(correlationPower);

% 4. Numerical protection
if maxPower > 0
    M = correlationPower / maxPower;
else
    M = zeros(size(correlationPower));
end

end