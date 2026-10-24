function [P, R, M] = schmidl_cox_metric(rxSignal, L)
% SCHMIDL_COX_METRIC
% Calculate Schmidl-Cox timing metric.
% Inputs:
%  rxSignal - Received complex baseband signal
%  L - Half of the useful OFDM symbol length
% Outputs:
%  P - Delayed correlation
%  R - Energy normalization
%  M - Normalized timing metric

%% 1. Input Preparation
rxSignal = rxSignal(:);

numCandidates = length(rxSignal) - 2*L + 1;

if numCandidates < 1
    error('Received signal is too short.');
end

%% 2. Initialization
P = zeros(numCandidates, 1);
R = zeros(numCandidates, 1);
M = zeros(numCandidates, 1);

%% 3. Timing Metric Calculation
for d = 1:numCandidates

    r1 = rxSignal(d:d+L-1);

    r2 = rxSignal(d+L:d+2*L-1);

    P(d) = sum(conj(r1) .* r2);

    R(d) = sum(abs(r2).^2);

    if R(d) > 1e-12

        M(d) = abs(P(d))^2 / R(d)^2;

    else

        M(d) = 0;

    end

end

end