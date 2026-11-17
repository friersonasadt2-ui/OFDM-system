%% EXP04 - Park Peak Selection Strategy Comparison
% Noiseless paired comparison
%
% Compare:
%   1. Park 2003 Metric + Maximum Peak Selection
%   2. Park 2003 Metric + Early Peak Selection
%
% Both estimators use exactly the same received signal.
%
% NOTE:
% The search window, energy gate, peak selection strategy,
% and center-to-start mapping are experimental choices.
% The Early Peak method is NOT the original Park 2003 algorithm.

clearvars;
clc;

%% 1. Project setup
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(2027);

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

%% 2. Experiment parameters
numTrials = 100;

arrivalRange = [20 80];

energyThreshold = 8;
metricThreshold = 0.95;

errorsMax = NaN(numTrials,1);
errorsEarly = NaN(numTrials,1);

%% 3. Monte Carlo simulation
for t = 1:numTrials

    % Random arrival delay
    D = randi(arrivalRange);

    % Original Park frequency-domain training construction
    pn = 2*randi([0 1],L,1)-1;

    X = zeros(N,1);
    X(1:2:end) = pn;

    x = ifft(X,N);

    training = [x(end-Ncp+1:end);x];

    training = normalize_training_energy( ...
        training,cfg.trainingEnergy);

    % Random QPSK OFDM data symbol
    dataIdx = randi([0 3],N,1);

    dataFreq = pskmod(dataIdx,4,pi/4);
    dataTime = ifft(dataFreq,N);

    dataSymbol = ...
        [dataTime(end-Ncp+1:end);dataTime];

    dataSymbol = normalize_training_energy( ...
        dataSymbol,cfg.trainingEnergy);

    % Same received frame for both estimators
    txFrame = [training;dataSymbol];

    rx = [zeros(D,1);txFrame;zeros(N,1)];

    trueStart = D+Ncp+1;

    % Estimator 1: Maximum Peak
    [estMax,~,~,validMax] = ...
        park_estimator_paper2003( ...
        rx,cfg,arrivalRange,energyThreshold);

    % Estimator 2: Earliest Reliable Peak
    [estEarly,~,~,validEarly] = ...
        park_estimator_early_peak( ...
        rx,cfg,arrivalRange, ...
        energyThreshold,metricThreshold);

    % Record timing errors

    if validMax
        errorsMax(t) = estMax-trueStart;
    end

    if validEarly
        errorsEarly(t) = estEarly-trueStart;
    end

end

%% 4. Performance statistics
% Exact timing probability
% Denominator includes all trials, including no-estimate cases.

exactMax = mean(errorsMax==0);
exactEarly = mean(errorsEarly==0);

% CP-safe timing probability
% A timing estimate within [-Ncp, 0] is considered CP-safe
% under the current experiment's definition.

safeMax = mean(errorsMax>=-Ncp & errorsMax<=0);
safeEarly = mean(errorsEarly>=-Ncp & errorsEarly<=0);

% No-estimate probability

failMax = mean(isnan(errorsMax));
failEarly = mean(isnan(errorsEarly));

% Conditional MSE (valid estimates only)

validErrorsMax = errorsMax(~isnan(errorsMax));
validErrorsEarly = errorsEarly(~isnan(errorsEarly));

if isempty(validErrorsMax)
    mseMax = NaN;
else
    mseMax = mean(validErrorsMax.^2);
end

if isempty(validErrorsEarly)
    mseEarly = NaN;
else
    mseEarly = mean(validErrorsEarly.^2);
end

%% 5. Print performance comparison
fprintf('\nPark Peak Strategy Comparison\n');
fprintf('Trials = %d\n',numTrials);

fprintf('Method\t\tExact\tCP-safe\tFail\tCond.MSE\n');

fprintf('Max Peak\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
    exactMax,safeMax,failMax,mseMax);

fprintf('Early Peak\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
    exactEarly,safeEarly,failEarly,mseEarly);

%% 6. Maximum Peak error distribution
fprintf('\nMaximum Peak Error Distribution\n');

if isempty(validErrorsMax)

    fprintf('No valid estimates.\n');

else

    [uniqueErrors,~,groupIdx] = unique(validErrorsMax);

    counts = accumarray(groupIdx,1);

    for k = 1:numel(uniqueErrors)

        fprintf('Error %+d: %d trials\n', ...
            uniqueErrors(k),counts(k));

    end

end

%% 7. Early Peak error distribution
fprintf('\nEarly Peak Error Distribution\n');

if isempty(validErrorsEarly)

    fprintf('No valid estimates.\n');

else

    [uniqueErrors,~,groupIdx] = unique(validErrorsEarly);

    counts = accumarray(groupIdx,1);

    for k = 1:numel(uniqueErrors)

        fprintf('Error %+d: %d trials\n', ...
            uniqueErrors(k),counts(k));

    end

end

%% 8. Summary
fprintf('\nExperiment completed successfully.\n');