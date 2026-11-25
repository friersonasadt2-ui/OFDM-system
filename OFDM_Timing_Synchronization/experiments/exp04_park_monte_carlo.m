%% EXP04 Park Formal Monte Carlo
% Supervisor-confirmed Park implementation
% Paired AWGN and fixed multipath experiments

clearvars; clc; close all;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2040;
rng(randomSeed);

%% 1. Experiment configuration

N = cfg.Nfft;
Ncp = cfg.Ncp;

arrivalRange = [20 80];
energyThreshold = 8;

snrList = cfg.SNR_dB(:);
numTrials = cfg.numTrials;

numSNR = numel(snrList);

channelNames = {'AWGN','Fixed_Multipath'};

channelResponses = {1,cfg.hMultipath(:)};

numChannels = numel(channelNames);

maxPathDelays = [0,numel(cfg.hMultipath)-1];

cpSafeMin = -Ncp + maxPathDelays;

assert(Ncp>=max(maxPathDelays));

assert(abs(sum(abs(cfg.hMultipath).^2)-1)<1e-10);

%% 2. Initialize trial-level results

allErrors = NaN(numTrials,numSNR,numChannels);

allValid = false(numTrials,numSNR,numChannels);

allPeakMetrics = NaN(numTrials,numSNR,numChannels);

allSignalPowers = NaN(numTrials,numSNR,numChannels);

allArrivalDelays = zeros(numTrials,numSNR);

%% 3. Initialize summary statistics

exactProbability = zeros(numSNR,numChannels);

cpSafeProbability = zeros(numSNR,numChannels);

failProbability = zeros(numSNR,numChannels);

conditionalMSE = NaN(numSNR,numChannels);

meanError = NaN(numSNR,numChannels);

errorStd = NaN(numSNR,numChannels);

maxAbsError = NaN(numSNR,numChannels);

meanSignalPower = NaN(numSNR,numChannels);

correctCount = zeros(numSNR,numChannels);

failCount = zeros(numSNR,numChannels);

%% 4. Display experiment configuration

fprintf('EXP04 Park Formal Monte Carlo\n');

fprintf('Random Seed = %d\n',randomSeed);

fprintf('FFT Size = %d\n',N);

fprintf('CP Length = %d\n',Ncp);

fprintf('Training Energy = %.2f\n',cfg.trainingEnergy);

fprintf('Trials per SNR = %d\n',numTrials);

fprintf('Total Trials = %d\n', ...
    numTrials*numSNR*numChannels);

fprintf('Arrival Range = [%d, %d]\n',arrivalRange);

fprintf('Energy Threshold = %.2f\n',energyThreshold);

fprintf('Multipath Taps = ');
fprintf('%.6f ',cfg.hMultipath);
fprintf('\n');

fprintf('AWGN CP-safe Range = [%d, 0]\n',cpSafeMin(1));

fprintf('Multipath CP-safe Range = [%d, 0]\n\n',cpSafeMin(2));

%% 5. Monte Carlo simulation

tic;

for s = 1:numSNR

    snrDb = snrList(s);

    for trial = 1:numTrials

        % Random arrival delay
        D = randi(arrivalRange);

        allArrivalDelays(trial,s) = D;

        % Generate Park training symbol
        [training,~] = park_training_original(cfg);

        % Generate QPSK data symbol
        dataIdx = randi([0 3],N,1);

        dataFreq = qammod(dataIdx,4, ...
            'UnitAveragePower',true);

        xData = ifft(dataFreq,N);

        dataSymbol = ...
            [xData(end-Ncp+1:end);xData];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        % Verify transmission energy
        assert(abs(sum(abs(training).^2) ...
            -cfg.trainingEnergy)<1e-10);

        assert(abs(sum(abs(dataSymbol).^2) ...
            -cfg.trainingEnergy)<1e-10);

        % Construct transmitted frame
        txFrame = [training;dataSymbol];

        txBuffered = [zeros(D,1); ...
                      txFrame;zeros(N,1)];

        trueStart = D+Ncp+1;

        % Generate common unit-power complex Gaussian noise
        maxRxLength = numel(txBuffered) ...
            +max(maxPathDelays);

        baseNoise = ...
            (randn(maxRxLength,1) ...
            +1j*randn(maxRxLength,1))/sqrt(2);

        % Process both channel conditions
        for c = 1:numChannels

            h = channelResponses{c};

            rxClean = conv(txBuffered,h);

            % Active received frame power
            activeFrame = ...
                rxClean(D+1:D+numel(txFrame));

            signalPower = mean(abs(activeFrame).^2);

            assert(isfinite(signalPower) ...
                && signalPower>0);

            allSignalPowers(trial,s,c) = signalPower;

            % Set target received sample SNR
            noisePower = ...
                signalPower/10^(snrDb/10);

            noise = sqrt(noisePower) ...
                *baseNoise(1:numel(rxClean));

            rx = rxClean + noise;

            % Park timing estimation
            [estimatedStart,~,peakMetric,isValid] = ...
                park_estimator_paper2003( ...
                rx,cfg,arrivalRange,energyThreshold);

            allValid(trial,s,c) = isValid;

            if isValid

                assert(isfinite(estimatedStart));

                allErrors(trial,s,c) = ...
                    estimatedStart-trueStart;

                allPeakMetrics(trial,s,c) = ...
                    peakMetric;

            end

        end

    end

    %% 6. Calculate statistics at current SNR

    for c = 1:numChannels

        errors = allErrors(:,s,c);

        validErrors = errors(isfinite(errors));

        correctCount(s,c) = sum(errors==0);

        failCount(s,c) = sum(~allValid(:,s,c));

        exactProbability(s,c) = ...
            mean(errors==0);

        cpSafeProbability(s,c) = ...
            mean(errors>=cpSafeMin(c) ...
            & errors<=0);

        failProbability(s,c) = ...
            failCount(s,c)/numTrials;

        meanSignalPower(s,c) = ...
            mean(allSignalPowers(:,s,c));

        if ~isempty(validErrors)

            conditionalMSE(s,c) = ...
                mean(validErrors.^2);

            meanError(s,c) = ...
                mean(validErrors);

            errorStd(s,c) = ...
                std(validErrors,1);

            maxAbsError(s,c) = ...
                max(abs(validErrors));

        end

    end

    fprintf('SNR = %2d dB completed | ',snrDb);

    fprintf('AWGN Exact = %.4f | ', ...
        exactProbability(s,1));

    fprintf('Multipath Exact = %.4f\n', ...
        exactProbability(s,2));

end

elapsedTime = toc;

%% 7. Statistical consistency checks

assert(all(exactProbability(:) ...
    <=cpSafeProbability(:)));

assert(all(cpSafeProbability(:) ...
    <=1-failProbability(:)+1e-12));

assert(all(isfinite(allErrors(allValid))));

assert(all(isnan(allErrors(~allValid))));

assert(all(isfinite(allSignalPowers(:))));

assert(all(allSignalPowers(:)>0));

assert(all(abs(allSignalPowers(:,:,1)-1)<1e-10,'all'));

assert(all(abs( ...
    exactProbability(:)-correctCount(:)/numTrials) ...
    <1e-12));

%% 8. Create summary table

resultsTable = table( ...
    snrList, ...
    exactProbability(:,1), ...
    cpSafeProbability(:,1), ...
    failProbability(:,1), ...
    conditionalMSE(:,1), ...
    exactProbability(:,2), ...
    cpSafeProbability(:,2), ...
    failProbability(:,2), ...
    conditionalMSE(:,2), ...
    correctCount(:,1), ...
    correctCount(:,2), ...
    failCount(:,1), ...
    failCount(:,2), ...
    meanError(:,1), ...
    meanError(:,2), ...
    errorStd(:,1), ...
    errorStd(:,2), ...
    maxAbsError(:,1), ...
    maxAbsError(:,2), ...
    meanSignalPower(:,1), ...
    meanSignalPower(:,2), ...
    'VariableNames',{ ...
    'SNR_dB', ...
    'AWGN_Exact', ...
    'AWGN_CPsafe', ...
    'AWGN_Fail', ...
    'AWGN_MSE', ...
    'MP_Exact', ...
    'MP_CPsafe', ...
    'MP_Fail', ...
    'MP_MSE', ...
    'AWGN_CorrectCount', ...
    'MP_CorrectCount', ...
    'AWGN_FailCount', ...
    'MP_FailCount', ...
    'AWGN_MeanError', ...
    'MP_MeanError', ...
    'AWGN_ErrorStd', ...
    'MP_ErrorStd', ...
    'AWGN_MaxAbsError', ...
    'MP_MaxAbsError', ...
    'AWGN_SignalPower', ...
    'MP_SignalPower'});

%% 9. Save results

dataDir = fullfile(projectRoot,'results','data');

tableDir = fullfile(projectRoot,'results','tables');

figureDir = fullfile(projectRoot,'results','figures');

if ~exist(dataDir,'dir'); mkdir(dataDir); end

if ~exist(tableDir,'dir'); mkdir(tableDir); end

if ~exist(figureDir,'dir'); mkdir(figureDir); end

algorithmVersion = ...
    'Supervisor-confirmed mirror correlation, November 2026';

save(fullfile(dataDir,'exp04_park_monte_carlo.mat'), ...
    'cfg','randomSeed','algorithmVersion', ...
    'snrList','numTrials', ...
    'arrivalRange','energyThreshold', ...
    'channelNames','channelResponses', ...
    'maxPathDelays','cpSafeMin', ...
    'allArrivalDelays','allErrors', ...
    'allValid','allPeakMetrics','allSignalPowers', ...
    'exactProbability','cpSafeProbability', ...
    'failProbability','conditionalMSE', ...
    'meanError','errorStd','maxAbsError', ...
    'meanSignalPower','correctCount','failCount', ...
    'resultsTable','elapsedTime');

writetable(resultsTable, ...
    fullfile(tableDir,'exp04_park_monte_carlo.csv'));

%% 10. Plot and export results

metricData = { ...
    conditionalMSE, ...
    exactProbability, ...
    cpSafeProbability, ...
    failProbability};

metricTitles = { ...
    'Park Timing MSE vs SNR', ...
    'Park Exact Timing Probability vs SNR', ...
    'Park CP-safe Probability vs SNR', ...
    'Park Failure Probability vs SNR'};

yLabels = { ...
    'Conditional Timing MSE (samples^2)', ...
    'Exact Timing Probability', ...
    'CP-safe Probability', ...
    'Failure Probability'};

figureNames = { ...
    'exp04_park_mse.png', ...
    'exp04_park_exact.png', ...
    'exp04_park_cpsafe.png', ...
    'exp04_park_fail.png'};

for k = 1:numel(metricData)

    fig = figure('Color','w');

    values = metricData{k};

    plot(snrList,values(:,1), ...
        'b-o','LineWidth',1.5, ...
        'MarkerSize',5);

    hold on;

    plot(snrList,values(:,2), ...
        'r-s','LineWidth',1.5, ...
        'MarkerSize',5);

    grid on;

    xlabel('SNR (dB)');

    ylabel(yLabels{k});

    title(metricTitles{k});

    legend('AWGN','Fixed Multipath + AWGN', ...
        'Location','best');

    xlim([min(snrList),max(snrList)]);

    xticks(snrList);

    if k~=1
        ylim([0 1.05]);
    end

    exportgraphics(fig, ...
        fullfile(figureDir,figureNames{k}), ...
        'Resolution',300);

end

%% 11. Display final summary

fprintf('\nFinal Park Results\n');

disp(resultsTable(:,1:9));

fprintf('Simulation Time = %.2f seconds\n', ...
    elapsedTime);

fprintf('\nSaved MAT:\n%s\n', ...
    fullfile(dataDir,'exp04_park_monte_carlo.mat'));

fprintf('\nSaved CSV:\n%s\n', ...
    fullfile(tableDir,'exp04_park_monte_carlo.csv'));

fprintf('\nSaved Figures:\n');

for k = 1:numel(figureNames)
    fprintf('%s\n',fullfile(figureDir,figureNames{k}));
end

fprintf('\nEXP04 Park Monte Carlo completed.\n');
