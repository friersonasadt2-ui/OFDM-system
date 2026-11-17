
function [P, R, M, centerIndex] = park_metric_paper2003(rx, Nfft)
% Park et al. (2003), IEEE Communications Letters
% Implements equations (11)-(13).
%
% d is the center sample index (MATLAB 1-based).
% This function calculates the paper's timing metric,
% not the final useful-symbol-start estimate.

rx = rx(:);

assert(mod(Nfft,2)==0, 'Nfft must be even.');

L = Nfft/2;
numSamples = numel(rx);

assert(numSamples >= 2*L+1, ...
    'Received signal is too short.');

% Valid centers for k = 0,...,L
centerIndex = (L+1:numSamples-L).';

P = zeros(size(centerIndex));
R = zeros(size(centerIndex));
M = zeros(size(centerIndex));

for i = 1:numel(centerIndex)

    d = centerIndex(i);

    k = (0:L).';

    left = rx(d-k);
    right = rx(d+k);

    % Equation (12)
    P(i) = sum(left .* right);

    % Equation (13)
    R(i) = sum(abs(right).^2);

    % Equation (11)
    if R(i) > 1e-12
        M(i) = abs(P(i))^2 / R(i)^2;
    else
        M(i) = 0;
    end
end
end
