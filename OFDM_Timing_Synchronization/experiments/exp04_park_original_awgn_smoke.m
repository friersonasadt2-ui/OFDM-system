%% EXP04 Park AWGN Smoke Test
% Supervisor-confirmed Park implementation
% Random training, data, arrival delay and AWGN

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

rng(2038);

N = cfg.Nfft;
Ncp = cfg.Ncp;

arrivalRange = [20 80];
energyThreshold = 8;

snrList = [0 10 20];
numTrials = 100;

numSNR = numel(snrList);

%% Initialize results

allErrors = NaN(numTrials,numSNR);
allValid = false(numTrials,numSNR);
allPeakMetrics = NaN(numTrials,numSNR);

exactProbability = zeros(numSNR,1);
cpSafeProbability = zeros(numSNR,1);
failProbability = zeros(numSNR,1);
conditionalMSE = NaN(numSNR,1);
meanError = NaN(numSNR,1);
maxAbsError = NaN(numSNR,1);

fprintf('Park AWGN Smoke Test\n');
fprintf('Random Seed = 2038\n');
fprintf('Trials per SNR = %d\n',numTrials);
fprintf('Arrival Range = [%d, %d]\n',arrivalRange);
fprintf('Energy Threshold = %.2f\n\n',energyThreshold);

%% Monte Carlo simulation

tic;

for s = 1:numSNR

    snrDb = snrList(s);

    for trial = 1:numTrials

        % 1. Random arrival delay
        D = randi(arrivalRange);

        % 2. Park training symbol
        [training,~] = park_training_original(cfg);

        % 3. QPSK data symbol
        dataIdx = randi([0 3],N,1);

        dataFreq = qammod(dataIdx,4, ...
            'UnitAveragePower',true);

        xData = ifft(dataFreq,N);

        dataSymbol = ...
            [xData(end-Ncp+1:end);xData];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        % 4. Construct transmitted frame
        txFrame = [training;dataSymbol];

        rxClean = [zeros(D,1); ...
                   txFrame;zeros(N,1)];

        % 5. Verify energy and signal power
        assert(abs(sum(abs(training).^2) ...
            -cfg.trainingEnergy)<1e-10);

        assert(abs(sum(abs(dataSymbol).^2) ...
            -cfg.trainingEnergy)<1e-10);

        signalPower = mean(abs(txFrame).^2);

        assert(abs(signalPower-1)<1e-10);

        % 6. Add complex AWGN
        noisePower = signalPower / 10^(snrDb/10);

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean + noise;

        % 7. True useful FFT start
        trueStart = D+Ncp+1;

        % 8. Park timing estimation
        [estimatedStart,~,peakMetric,isValid] = ...
            park_estimator_paper2003( ...
            rx,cfg,arrivalRange,energyThreshold);

        % 9. Record trial results
        allValid(trial,s) = isValid;

        if isValid

            assert(isfinite(estimatedStart));

            allErrors(trial,s) = ...
                estimatedStart-trueStart;

            allPeakMetrics(trial,s) = peakMetric;

        end

    end

    %% Calculate statistics for current SNR

    errors = allErrors(:,s);
    validErrors = errors(isfinite(errors));

    exactProbability(s) = mean(errors==0);

    cpSafeProbability(s) = ...
        mean(errors>=-Ncp & errors<=0);

    failProbability(s) = mean(~allValid(:,s));

    if ~isempty(validErrors)

        conditionalMSE(s) = mean(validErrors.^2);

        meanError(s) = mean(validErrors);

        maxAbsError(s) = max(abs(validErrors));

    end

end

elapsedTime = toc;

%% Display results

fprintf('SNR(dB)    Exact    CP-safe    Fail    Cond.MSE\n');

for s = 1:numSNR

    fprintf('%7.0f %9.4f %10.4f %8.4f %11.4f\n', ...
        snrList(s), ...
        exactProbability(s), ...
        cpSafeProbability(s), ...
        failProbability(s), ...
        conditionalMSE(s));

end

%% Timing error distribution

for s = 1:numSNR

    fprintf('\nSNR = %d dB\n',snrList(s));

    fprintf('Correct Estimates = %d / %d\n', ...
        sum(allErrors(:,s)==0),numTrials);

    fprintf('Failed Estimates = %d\n', ...
        sum(~allValid(:,s)));

    fprintf('Mean Timing Error = %+.4f\n', ...
        meanError(s));

    fprintf('Maximum Absolute Error = %.0f\n', ...
        maxAbsError(s));

    errors = allErrors(:,s);
    errors = errors(isfinite(errors));

    uniqueErrors = unique(errors);

    for k = 1:numel(uniqueErrors)

        fprintf('Error %+g: %d trials\n', ...
            uniqueErrors(k), ...
            sum(errors==uniqueErrors(k)));

    end

end

%% Basic statistical consistency checks

assert(all(exactProbability<=cpSafeProbability));

assert(all(cpSafeProbability<=1-failProbability+1e-12));

assert(all(isnan(allErrors(~allValid))));

assert(all(isfinite(allErrors(allValid))));

fprintf('\nSignal Power = %.6f\n',signalPower);

fprintf('Total Trials = %d\n',numTrials*numSNR);

fprintf('Elapsed Time = %.2f seconds\n',elapsedTime);

fprintf('Park AWGN smoke test completed.\n');
