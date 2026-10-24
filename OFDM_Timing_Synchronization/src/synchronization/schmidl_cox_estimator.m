function [estimatedStart, info] = schmidl_cox_estimator(M, alpha)
% SCHMIDL_COX_ESTIMATOR
% Estimate timing position using relative threshold
% and plateau midpoint detection.
% Inputs:
%  M - Schmidl-Cox timing metric
%  alpha - Relative threshold factor
% Outputs:
%  estimatedStart - Estimated timing position
%   nfo - Detection information

%% 1. Input Preparation
M = M(:);

if isempty(M) || any(~isfinite(M))
    error('Timing metric must be nonempty and finite.');
end

if alpha <= 0 || alpha >= 1
    error('Alpha must be between 0 and 1.');
end

%% 2. Find Maximum Timing Metric
[maxMetric, maxIndex] = max(M);

% Reject signals without a positive timing metric
if maxMetric <= 0
    estimatedStart = NaN;

    info.maxMetric = maxMetric;
    info.maxIndex = maxIndex;
    info.threshold = NaN;
    info.plateauStart = NaN;
    info.plateauEnd = NaN;
    info.detected = false;

    return;
end

%% 3. Relative Threshold
threshold = alpha * maxMetric;

aboveThreshold = (M >= threshold);

%% 4. Detect Continuous Regions
edges = diff([false; aboveThreshold; false]);

regionStarts = find(edges == 1);
regionEnds = find(edges == -1) - 1;

%% 5. Select Region Containing Maximum
selectedRegion = find( ...
    regionStarts <= maxIndex & ...
    regionEnds >= maxIndex, 1);

plateauStart = regionStarts(selectedRegion);
plateauEnd = regionEnds(selectedRegion);

%% 6. Estimate Timing Position
estimatedStart = round( ...
    (plateauStart + plateauEnd) / 2);

%% 7. Return Detection Information
info.maxMetric = maxMetric;
info.maxIndex = maxIndex;
info.threshold = threshold;
info.plateauStart = plateauStart;
info.plateauEnd = plateauEnd;
info.detected = true;

end