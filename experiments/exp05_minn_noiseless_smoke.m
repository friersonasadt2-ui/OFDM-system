%% EXP05 - Minn Random Noiseless Test
clear; clc;

cfg = default_config();
rng(2026);

N = cfg.Nfft;
Ncp = cfg.Ncp;
numTrials = 100;

errors = NaN(numTrials,1);
peakMetrics = NaN(numTrials,1);
estimatedStarts = NaN(numTrials,1);
trueStarts = zeros(numTrials,1);
validFlags = false(numTrials,1);

for t = 1:numTrials

    % 1. Generate Minn training
    [training,~,~] = minn_training(cfg);

    % 2. Generate new random QPSK data
    dataIndices = randi([0 3],N,1);
    dataQPSK = pskmod(dataIndices,4,pi/4);

    dataTime = ifft(dataQPSK,N);
    dataSymbol = [dataTime(end-Ncp+1:end);dataTime];

    dataSymbol = normalize_training_energy( ...
        dataSymbol,cfg.trainingEnergy);

    % 3. Random arrival delay
    D = randi([20 80]);
    trueStart = D+Ncp+1;

    txFrame = [training;dataSymbol];

    rx = [zeros(D,1);txFrame;zeros(N,1)];

    % 4. Run Minn estimator
    [estimatedStart,peakMetric,isValid,~] = ...
        minn_estimator(rx,cfg);

    trueStarts(t) = trueStart;
    estimatedStarts(t) = estimatedStart;
    peakMetrics(t) = peakMetric;
    validFlags(t) = isValid;

    if isValid
        errors(t) = estimatedStart-trueStart;
    end

end

%% 5. Calculate statistics
validErrors = errors(isfinite(errors));

exactProb = mean(errors==0);
cpSafeProb = mean(errors>=-Ncp & errors<=0);
failProb = mean(~validFlags);

if isempty(validErrors)
    condMSE = NaN;
else
    condMSE = mean(validErrors.^2);
end

fprintf('\nEXP05 Minn Random Noiseless Test\n');
fprintf('Trials = %d\n',numTrials);
fprintf('Exact Probability = %.4f\n',exactProb);
fprintf('CP-safe Probability = %.4f\n',cpSafeProb);
fprintf('Failure Probability = %.4f\n',failProb);
fprintf('Conditional Timing MSE = %.4f\n',condMSE);

%% 6. Display error distribution
fprintf('\nTiming Error Distribution:\n');

if ~isempty(validErrors)
    [uniqueErrors,~,idx] = unique(validErrors);
    counts = accumarray(idx,1);

    for k = 1:numel(uniqueErrors)
        fprintf('Error %+d : %d trials\n', ...
            uniqueErrors(k),counts(k));
    end
end

%% 7. Report incorrect estimates
wrong = find(~isfinite(errors) | errors~=0);

fprintf('\nIncorrect/Failed Trials = %d\n',numel(wrong));

for k = 1:numel(wrong)
    t = wrong(k);
    fprintf('Trial %d: True=%d, Estimated=%.0f, Error=%+.0f\n', ...
        t,trueStarts(t),estimatedStarts(t),errors(t));
end

disp('EXP05 Minn noiseless smoke test completed.');
