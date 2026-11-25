%% EXP04 Park Multipath Smoke Test
% Supervisor-confirmed Park implementation
% Fixed three-path channel + AWGN

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

rng(2039);

N = cfg.Nfft;
Ncp = cfg.Ncp;

arrivalRange = [20 80];
energyThreshold = 8;

snrList = [0 10 20];
numTrials = 100;
numSNR = numel(snrList);

%% Channel configuration

h = cfg.hMultipath(:);

maxPathDelay = numel(h)-1;

cpSafeMin = -(Ncp-maxPathDelay);

assert(abs(sum(abs(h).^2)-1)<1e-10);
assert(Ncp>=maxPathDelay);

%% Initialize results

allErrors = NaN(numTrials,numSNR);
allValid = false(numTrials,numSNR);
allPeakMetrics = NaN(numTrials,numSNR);
allSignalPowers = NaN(numTrials,numSNR);

exactProbability = zeros(numSNR,1);
cpSafeProbability = zeros(numSNR,1);
failProbability = zeros(numSNR,1);
conditionalMSE = NaN(numSNR,1);

meanError = NaN(numSNR,1);
maxAbsError = NaN(numSNR,1);

fprintf('Park Multipath Smoke Test\n');

fprintf('Random Seed = 2039\n');
fprintf('Trials per SNR = %d\n',numTrials);

fprintf('Channel Taps = ');
fprintf('%.6f ',h);
fprintf('\n');

fprintf('Channel Energy = %.8f\n',sum(abs(h).^2));
fprintf('Maximum Path Delay = %d\n',maxPathDelay);
fprintf('CP-safe Range = [%d, 0]\n',cpSafeMin);
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

        % 4. Verify symbol energies
        assert(abs(sum(abs(training).^2) ...
            -cfg.trainingEnergy)<1e-10);

        assert(abs(sum(abs(dataSymbol).^2) ...
            -cfg.trainingEnergy)<1e-10);

        % 5. Construct transmitted frame
        txFrame = [training;dataSymbol];

        txBuffered = [zeros(D,1); ...
                      txFrame;zeros(N,1)];

        % 6. Apply fixed multipath channel
        rxClean = conv(txBuffered,h);

        assert(numel(rxClean)== ...
            numel(txBuffered)+numel(h)-1);

        % 7. Measure active received frame power
        activeStart = D+1;
        activeEnd = D+numel(txFrame);

        activeFrame = ...
            rxClean(activeStart:activeEnd);

        signalPower = mean(abs(activeFrame).^2);

        assert(isfinite(signalPower));
        assert(signalPower>0);

        allSignalPowers(trial,s) = signalPower;

        % 8. Add complex AWGN after multipath
        noisePower = signalPower / 10^(snrDb/10);

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean + noise;

        % 9. First-path useful FFT reference
        trueStart = D+Ncp+1;

        % 10. Park timing estimation
        [estimatedStart,~,peakMetric,isValid] = ...
            park_estimator_paper2003( ...
            rx,cfg,arrivalRange,energyThreshold);

        allValid(trial,s) = isValid;

        if isValid

            assert(isfinite(estimatedStart));

            allErrors(trial,s) = ...
                estimatedStart-trueStart;

            allPeakMetrics(trial,s) = peakMetric;

        end

    end

    %% Calculate performance statistics

    errors = allErrors(:,s);

    validErrors = errors(isfinite(errors));

    exactProbability(s) = mean(errors==0);

    cpSafeProbability(s) = mean( ...
        errors>=cpSafeMin & errors<=0);

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

    fprintf('Mean Active Received Power = %.6f\n', ...
        mean(allSignalPowers(:,s)));

    errors = allErrors(:,s);
    errors = errors(isfinite(errors));

    uniqueErrors = unique(errors);

    for k = 1:numel(uniqueErrors)

        fprintf('Error %+g: %d trials\n', ...
            uniqueErrors(k), ...
            sum(errors==uniqueErrors(k)));

    end

end

%% Statistical consistency checks

assert(all(exactProbability<=cpSafeProbability));

assert(all(cpSafeProbability<= ...
    1-failProbability+1e-12));

assert(all(isnan(allErrors(~allValid))));

assert(all(isfinite(allErrors(allValid))));

assert(all(isfinite(allSignalPowers(:))));

fprintf('\nTotal Trials = %d\n',numTrials*numSNR);

fprintf('Elapsed Time = %.2f seconds\n',elapsedTime);

fprintf('Park multipath smoke test completed.\n');
