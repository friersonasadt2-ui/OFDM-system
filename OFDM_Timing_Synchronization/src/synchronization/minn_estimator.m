function [estimatedStart, peakMetric, isValid, peakIndex] = ...
    minn_estimator(rx, cfg)
% MINN_ESTIMATOR
% Minn timing estimation using maximum metric.
% Inputs:
% rx: received sample buffer
% cfg: OFDM configuration
% Outputs:
% estimatedStart : estimated useful-symbol start
% peakMetric: maximum timing metric
% isValid: whether an estimate was returned
% peakIndex: selected metric candidate index
% No true timing position is used by this estimator.

% 1. Calculate timing metric
[~, ~, metric, candidateIndex] = ...
    minn_metric(rx, cfg.Nfft);

% 2. Initialize outputs
estimatedStart = NaN;
peakMetric = NaN;
peakIndex = NaN;
isValid = false;

% 3. Search all complete candidate windows
valid = isfinite(metric) & metric > 0;

if ~any(valid)
    return;
end

% 4. Select maximum metric
validIndices = find(valid);

[peakMetric, localIdx] = max(metric(valid));

selectedIdx = validIndices(localIdx);

peakIndex = candidateIndex(selectedIdx);

% 5. The metric candidate is the useful-symbol start
estimatedStart = peakIndex;

isValid = true;

end
