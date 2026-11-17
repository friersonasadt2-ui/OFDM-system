function [P, E, metric, candidateIndex] = ...
    cazac_metric(rx, training)
% CAZAC_METRIC
% Normalized matched-correlation timing metric.
% rx: received sample buffer
% training: known CAZAC symbol including CP
% candidateIndex represents the training start,
% NOT the useful FFT window start.

rx = rx(:);
training = training(:);

L = numel(training);

assert(numel(rx)>=L, ...
    'Received buffer is shorter than training.');

% Known training energy
trainingEnergy = sum(abs(training).^2);

assert(trainingEnergy>0, ...
    'Training energy must be positive.');

% Candidate training start positions
candidateIndex = (1:numel(rx)-L+1).';

numCandidates = numel(candidateIndex);

P = zeros(numCandidates,1);
E = zeros(numCandidates,1);
metric = zeros(numCandidates,1);

for i = 1:numCandidates

    d = candidateIndex(i);

    % Extract an 80-sample candidate window
    w = rx(d:d+L-1);

    % Matched correlation
    P(i) = sum(conj(training).*w);

    % Received window energy
    E(i) = sum(abs(w).^2);

    % Normalized correlation metric
    if E(i)>1e-12
        metric(i) = ...
            abs(P(i))^2/(trainingEnergy*E(i));
    else
        metric(i) = 0;
    end

end

end
