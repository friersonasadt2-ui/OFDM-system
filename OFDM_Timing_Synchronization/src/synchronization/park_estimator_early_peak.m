function [estimatedStart, peakCenter, peakMetric, isValid] = ...
    park_estimator_early_peak(rx, cfg, arrivalRange, ...
    energyThreshold, metricThreshold)
% Experimental Park early-reliable-peak estimator.
% NOT the original Park (2003) decision rule.

N = cfg.Nfft;
L = N/2;
rx = rx(:);

[~,~,M,centers] = park_metric_paper2003(rx,N);

searchMin = arrivalRange(1)+L+1;
searchMax = arrivalRange(2)+N+1;

valid = false(size(centers));

for i = 1:numel(centers)

    d = centers(i);

    if d < searchMin || d > searchMax
        continue;
    end

    k = (0:L).';

    left = rx(d-k);
    right = rx(d+k);

    EL = sum(abs(left).^2);
    ER = sum(abs(right).^2);

    valid(i) = ...
        EL >= energyThreshold && ...
        ER >= energyThreshold && ...
        M(i) >= metricThreshold;
end

estimatedStart = NaN;
peakCenter = NaN;
peakMetric = NaN;
isValid = false;

idx = find(valid,1,'first');

if isempty(idx)
    return;
end

peakCenter = centers(idx);
peakMetric = M(idx);

% Provisional experimental center-to-start mapping
estimatedStart = peakCenter-cfg.Ncp;

isValid = true;
end
