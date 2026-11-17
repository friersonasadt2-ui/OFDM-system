%% EXP06 - CAZAC Random Noiseless Test
clear; clc;

cfg = default_config();
rng(2026);

N = cfg.Nfft;
Ncp = cfg.Ncp;
numTrials = 100;

errors = NaN(numTrials,1);
trueStarts = zeros(numTrials,1);
estimatedStarts = NaN(numTrials,1);
peakMetrics = NaN(numTrials,1);
validFlags = false(numTrials,1);

% Fixed Zadoff-Chu training symbol
[training,~,~] = cazac_training(cfg);

for t = 1:numTrials

    % 1. Random QPSK OFDM data
    dataIndices = randi([0 3],N,1);
    dataQPSK = pskmod(dataIndices,4,pi/4);

    dataTime = ifft(dataQPSK,N);
    dataSymbol = ...
        [dataTime(end-Ncp+1:end);dataTime];

    dataSymbol = normalize_training_energy( ...
        dataSymbol,cfg.trainingEnergy);

    % 2. Random arrival delay
    D = randi([20 80]);

    trueTrainingStart = D+1;
    trueUsefulStart = D+Ncp+1;

    % 3. Continuous received buffer
    txFrame = [training;dataSymbol];
    rx = [zeros(D,1);txFrame;zeros(N,1)];

    % 4. CAZAC timing estimation
    [estimatedStart,peakMetric,isValid,peakIndex] = ...
        cazac_estimator(rx,cfg);

    trueStarts(t) = trueUsefulStart;
    estimatedStarts(t) = estimatedStart;
    peakMetrics(t) = peakMetric;
    validFlags(t) = isValid;

    if isValid && isfinite(estimatedStart)
        errors(t) = estimatedStart-trueUsefulStart;

        % Check index conversion consistency
        assert(estimatedStart == peakIndex+Ncp, ...
            'Training-to-FFT index conversion failed.');
    end

end

%% 5. Calculate statistics
validErrors = errors(isfinite(errors));

exactProb = mean(errors==0);
cpSafeProb = mean(errors>=-Ncp & errors<=0);
failProb = mean(~isfinite(errors));

if isempty(validErrors)
    condMSE = NaN;
else
    condMSE = mean(validErrors.^2);
end

fprintf('\nEXP06 CAZAC Random Noiseless Test\n');
fprintf('Trials = %d\n',numTrials);
fprintf('Exact Probability = %.4f\n',exactProb);
fprintf('CP-safe Probability = %.4f\n',cpSafeProb);
fprintf('Failure Probability = %.4f\n',failProb);
fprintf('Conditional Timing MSE = %.4f\n',condMSE);

%% 6. Error distribution
fprintf('\nTiming Error Distribution:\n');

if ~isempty(validErrors)
    [uniqueErrors,~,idx] = unique(validErrors);
    counts = accumarray(idx,1);

    for k = 1:numel(uniqueErrors)
        fprintf('Error %+d : %d trials\n', ...
            uniqueErrors(k),counts(k));
    end
end

%% 7. Incorrect/failed trials
wrong = find(~isfinite(errors) | errors~=0);

fprintf('\nIncorrect/Failed Trials = %d\n',numel(wrong));

for k = 1:numel(wrong)
    t = wrong(k);

    fprintf(['Trial %d: True=%d, Estimated=%.0f, ' ...
        'Error=%+.0f, Peak=%.6f\n'], ...
        t,trueStarts(t),estimatedStarts(t), ...
        errors(t),peakMetrics(t));
end

disp('EXP06 CAZAC noiseless smoke test completed.');