function [estimatedStart, peakCenter, peakMetric, ...
    isValid] = park_estimator_paper2003( ...
    rx, cfg, arrivalRange, energyThreshold)
% Park 2003 metric with experimental receiver selection.
%
% IMPORTANT:
% Metric: Park et al. (2003), equations (11)-(13).
% Search and energy gating: experimental receiver policy.
% Estimated-start mapping: provisional, not validated
% as the original Park paper's final decision rule.
%
% arrivalRange = [minimum possible delay, maximum delay]
% energyThreshold = minimum energy for each half-window

N = cfg.Nfft;
L = N/2;

assert(numel(arrivalRange)==2);
assert(arrivalRange(1)<=arrivalRange(2));
assert(energyThreshold>=0);

rx = rx(:);

% 1. Calculate the original Park timing metric
[~,~,M,centers] = park_metric_paper2003(rx,N);

% 2. Prespecified coarse timing search
% Diagnostic range established in previous tests
searchMin = arrivalRange(1)+L+1;
searchMax = arrivalRange(2)+N+1;

searchValid = ...
    centers>=searchMin & centers<=searchMax;

% 3. Energy validity checking
energyValid = false(size(centers));

for i = 1:numel(centers)

    if ~searchValid(i)
        continue;
    end

    d = centers(i);
    k = (0:L).';

    left = rx(d-k);
    right = rx(d+k);

    EL = sum(abs(left).^2);
    ER = sum(abs(right).^2);

    energyValid(i) = ...
        EL>=energyThreshold && ...
        ER>=energyThreshold;
end

% 4. Combine validity conditions
validCandidates = searchValid & energyValid;

estimatedStart = NaN;
peakCenter = NaN;
peakMetric = NaN;
isValid = false;

if ~any(validCandidates)
    return;
end

% 5. Select maximum valid timing metric
candidateCenters = centers(validCandidates);
candidateMetrics = M(validCandidates);

[peakMetric,idx] = max(candidateMetrics);

peakCenter = candidateCenters(idx);

% 6. Provisional timing-index conversion
estimatedStart = peakCenter-cfg.Ncp;

isValid = true;
end
