function [estimatedStart, peakMetric, isValid, info] = ...
    timing_estimator_unified(rx,cfg,algorithm,searchRange)
% TIMING_ESTIMATOR_UNIFIED
%
% Unified timing estimation interface for EXP07 and EXP08.
%
% All returned timing indices refer to the useful FFT start.
% The common search range is defined in FFT-start coordinates.
%
% Supported algorithms:
%   'SC', 'Park', 'Minn', 'CAZAC'
%
% No true arrival delay is used by the receiver.

rx = rx(:);

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

assert(numel(searchRange)==2);
assert(searchRange(1)<=searchRange(2));
assert(all(isfinite(rx)));

%% 1. Initialize outputs

estimatedStart = NaN;
peakMetric = NaN;
isValid = false;

info = struct();

info.algorithm = algorithm;
info.searchRange = searchRange;

%% 2. Calculate algorithm-specific metric

switch upper(char(algorithm))

    case 'SC'

        [~,~,metric] = ...
            schmidl_cox_metric(rx,L);

        rawIndex = (1:numel(metric)).';

        usefulIndex = rawIndex;

    case 'PARK'

        [~,~,metric,rawIndex] = ...
            park_metric_paper2003(rx,N);

        usefulIndex = rawIndex-L;

    case 'MINN'

        [~,~,metric,rawIndex] = ...
            minn_metric(rx,N);

        usefulIndex = rawIndex;

    case 'CAZAC'

        % Known deterministic ZC reference template
        training = cazac_training(cfg);

        [~,~,metric,rawIndex] = ...
            cazac_metric(rx,training);

        usefulIndex = rawIndex+Ncp;

    otherwise

        error('Unknown timing algorithm.');

end

%% 3. Convert to common search coordinates

metric = metric(:);
rawIndex = rawIndex(:);
usefulIndex = usefulIndex(:);

searchValid = ...
    usefulIndex>=searchRange(1) & ...
    usefulIndex<=searchRange(2);

metricValid = isfinite(metric) & metric>0;

valid = searchValid & metricValid;

%% 4. Retain Park energy gate

if strcmpi(algorithm,'Park')

    energyThreshold = 8;

    k = (1:L).';

    searchIndices = find(searchValid);

    for i = 1:numel(searchIndices)

        idx = searchIndices(i);

        d = rawIndex(idx);

        left = rx(d-k);

        right = rx(d+k-1);

        EL = sum(abs(left).^2);
        ER = sum(abs(right).^2);

        valid(idx) = valid(idx) && ...
            EL>=energyThreshold && ...
            ER>=energyThreshold;

    end

    info.energyThreshold = energyThreshold;

end

%% 5. Save diagnostic information

info.metric = metric;
info.rawIndex = rawIndex;
info.usefulIndex = usefulIndex;
info.searchValid = searchValid;
info.validCandidates = valid;

info.selectedRawIndex = NaN;
info.selectedUsefulIndex = NaN;

if ~any(valid)
    return;
end

%% 6. Algorithm-specific decision

if strcmpi(algorithm,'SC')

    % Preserve S&C relative threshold and
    % contiguous plateau midpoint rule.
    %
    % Keep the original index coordinates.
    % Candidates outside the common search
    % range are excluded by setting them to zero.

    restrictedMetric = zeros(size(metric));

    restrictedMetric(valid) = metric(valid);

    [localEstimate,scInfo] = ...
        schmidl_cox_estimator( ...
        restrictedMetric,cfg.SC_threshold);

    info.scInfo = scInfo;

    if ~scInfo.detected || ...
            ~isfinite(localEstimate)

        return;

    end

    estimatedStart = usefulIndex(localEstimate);

    peakMetric = scInfo.maxMetric;

    selectedIdx = localEstimate;

else

    % Park, Minn and CAZAC:
    % maximum positive valid timing metric

    validIndices = find(valid);

    [peakMetric,localIdx] = max(metric(valid));

    selectedIdx = validIndices(localIdx);

    estimatedStart = usefulIndex(selectedIdx);

end

%% 7. Validate final estimate

assert(isfinite(estimatedStart));

assert(estimatedStart>=searchRange(1));

assert(estimatedStart<=searchRange(2));

info.selectedRawIndex = rawIndex(selectedIdx);

info.selectedUsefulIndex = estimatedStart;

isValid = true;

end