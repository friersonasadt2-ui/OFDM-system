function [estimatedStart, peakMetric, isValid, peakIndex] = ...
    cazac_estimator(rx, cfg)
% CAZAC_ESTIMATOR
% ZC-based normalized matched-correlation estimator
% estimatedStart: FFT useful training start (1-based)
% peakMetric: selected maximum metric
% isValid: whether an estimate was returned
% peakIndex: estimated full training-symbol start
% No true timing position is used.
% 1. Generate known training sequence
[training,~,~] = cazac_training(cfg);

% 2. Calculate correlation metric
[~,~,metric,candidateIndex] = ...
    cazac_metric(rx,training);

% 3. Initialize outputs
estimatedStart = NaN;
peakMetric = NaN;
peakIndex = NaN;
isValid = false;

% 4. Find valid metric values
valid = isfinite(metric) & metric > 0;

if ~any(valid)
    return;
end

% 5. Select global maximum
validIndices = find(valid);

[peakMetric,localIdx] = max(metric(valid));

selectedIdx = validIndices(localIdx);

% Full training-symbol start
peakIndex = candidateIndex(selectedIdx);

% 6. Convert to useful FFT start
estimatedStart = peakIndex + cfg.Ncp;

isValid = true;

end