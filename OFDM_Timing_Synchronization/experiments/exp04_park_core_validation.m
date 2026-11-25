%% EXP04 Park Core Validation
% Supervisor-confirmed Park implementation
% No AWGN and no multipath

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

arrivalRange = [20 80];
energyThreshold = 8;

fprintf('Park Core Validation\n\n');

%% Test 1: Invalid correlation protection

rxZero = zeros(300,1);

% Set energy threshold to zero deliberately
% to isolate metric validity protection.
[est,peak,metric,valid] = ...
    park_estimator_paper2003( ...
    rxZero,cfg,arrivalRange,0);

assert(~valid);
assert(isnan(est));
assert(isnan(peak));
assert(isnan(metric));

fprintf('Test 1: Invalid Metric Protection\n');
fprintf('Valid Estimate = %d\n',valid);
fprintf('Test 1 PASSED\n\n');

%% Test 2: Reproduce supervisor's original experiment

rng(cfg.randomSeed);

D = cfg.arrivalDelay;

% IMPORTANT:
% Generate data first, then training,
% matching the supervisor's random-number order.

dataIdx = randi([0 3],N,1);
dataFreq = qammod(dataIdx,4, ...
    'UnitAveragePower',true);

xData = ifft(dataFreq,N);

[training,~] = park_training_original(cfg);

dataSymbol = [xData(end-Ncp+1:end);xData];

dataSymbol = normalize_training_energy( ...
    dataSymbol,cfg.trainingEnergy);

rx = [zeros(D,1);training; ...
      dataSymbol;zeros(N,1)];

trueStart = D+Ncp+1;
expectedCenter = trueStart+L;

[~,~,M,centers] = ...
    park_metric_paper2003(rx,N);

[sortedM,idx] = sort(M,'descend');

fprintf('Test 2: Supervisor Result Reproduction\n');

fprintf('True Start = %d\n',trueStart);
fprintf('Expected Center = %d\n',expectedCenter);

fprintf('Top 5 Metric Positions:\n');

for k = 1:5
    fprintf('  d=%d, M=%.8f\n', ...
        centers(idx(k)),sortedM(k));
end

[est,peak,metric,valid] = ...
    park_estimator_paper2003( ...
    rx,cfg,arrivalRange,energyThreshold);

fprintf('Estimated Start = %.0f\n',est);
fprintf('Peak Center = %.0f\n',peak);
fprintf('Peak Metric = %.8f\n',metric);
fprintf('Timing Error = %+g\n',est-trueStart);

assert(valid);
assert(est==trueStart);
assert(peak==expectedCenter);
assert(abs(metric-1)<1e-12);

referenceCenters = [99;67;115;107;142];

referenceMetrics = ...
    [1;0.25;0.15513835;0.11747856;0.08438875];

assert(isequal(centers(idx(1:5)),referenceCenters));
assert(max(abs(sortedM(1:5)-referenceMetrics))<1e-7);

fprintf('Test 2 PASSED\n\n');

%% Test 3: Randomized noiseless Monte Carlo

rng(2037);

numTrials = 100;

allErrors = NaN(numTrials,1);
allValid = false(numTrials,1);

for trial = 1:numTrials

    % Random arrival delay
    D = randi(arrivalRange);

    % Independent random training symbol
    [training,~] = park_training_original(cfg);

    % Independent random QPSK data symbol
    dataIdx = randi([0 3],N,1);

    dataFreq = qammod(dataIdx,4, ...
        'UnitAveragePower',true);

    xData = ifft(dataFreq,N);

    dataSymbol = [xData(end-Ncp+1:end);xData];

    dataSymbol = normalize_training_energy( ...
        dataSymbol,cfg.trainingEnergy);

    % Energy verification
    assert(abs(sum(abs(training).^2) ...
        -cfg.trainingEnergy)<1e-10);

    assert(abs(sum(abs(dataSymbol).^2) ...
        -cfg.trainingEnergy)<1e-10);

    % Construct noiseless received signal
    rx = [zeros(D,1);training; ...
          dataSymbol;zeros(N,1)];

    trueStart = D+Ncp+1;

    [est,~,~,valid] = ...
        park_estimator_paper2003( ...
        rx,cfg,arrivalRange,energyThreshold);

    allValid(trial) = valid;

    if valid
        allErrors(trial) = est-trueStart;
    end

end

%% Calculate performance statistics

finiteErrors = allErrors(isfinite(allErrors));

exactProbability = mean(allErrors==0);

cpSafeProbability = mean( ...
    allErrors>=-Ncp & allErrors<=0);

failProbability = mean(~allValid);

conditionalMSE = mean(finiteErrors.^2);

fprintf('Test 3: Randomized Noiseless Validation\n');
fprintf('Random Seed = 2037\n');
fprintf('Number of Trials = %d\n',numTrials);

fprintf('Valid Estimates = %d / %d\n', ...
    sum(allValid),numTrials);

fprintf('Correct Estimates = %d / %d\n', ...
    sum(allErrors==0),numTrials);

fprintf('Exact Probability = %.4f\n', ...
    exactProbability);

fprintf('CP-safe Probability = %.4f\n', ...
    cpSafeProbability);

fprintf('Fail Probability = %.4f\n', ...
    failProbability);

fprintf('Conditional MSE = %.8f\n', ...
    conditionalMSE);

fprintf('\nTiming Error Distribution:\n');

uniqueErrors = unique(finiteErrors);

for k = 1:numel(uniqueErrors)
    fprintf('Error %+g: %d trials\n', ...
        uniqueErrors(k), ...
        sum(finiteErrors==uniqueErrors(k)));
end

if all(allErrors==0)
    fprintf('\nAll Park core checks passed.\n');
else
    fprintf('\nRandomized validation requires review.\n');
end