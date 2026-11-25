%% EXP07 Unified Four-Algorithm Formal Monte Carlo
% Paired comparison of four OFDM timing estimators.
% AWGN and fixed three-path multipath channels.

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2045;
rng(randomSeed);

%% 1. Common experimental configuration

N = cfg.Nfft;
Ncp = cfg.Ncp;

arrivalRange = [20 80];

searchRange = [ ...
    arrivalRange(1)+1, ...
    arrivalRange(2)+2*Ncp+1];

snrList = cfg.SNR_dB(:);

numTrials = cfg.numTrials;

algorithmNames = {'SC','Park','Minn','CAZAC'};

channelNames = {'AWGN','Fixed_Multipath'};

channelResponses = {1,cfg.hMultipath(:)};

numAlgorithms = numel(algorithmNames);
numChannels = numel(channelNames);
numSNR = numel(snrList);

maxPathDelays = [0,numel(cfg.hMultipath)-1];

cpSafeMin = -Ncp+maxPathDelays;

bufferLength = arrivalRange(2) ...
    +2*(N+Ncp)+N+max(maxPathDelays);

%% 2. Initialize trial-level results

allErrors = NaN( ...
    numTrials,numSNR,numChannels,numAlgorithms);

allValid = false( ...
    numTrials,numSNR,numChannels,numAlgorithms);

allPeakMetrics = NaN(size(allErrors));

allSignalPowers = NaN(size(allErrors));

allArrivalDelays = zeros(numTrials,numSNR);

%% 3. Initialize summary statistics

exactProbability = zeros( ...
    numSNR,numChannels,numAlgorithms);

cpSafeProbability = zeros(size(exactProbability));

failProbability = zeros(size(exactProbability));

conditionalMSE = NaN(size(exactProbability));

meanError = NaN(size(exactProbability));

maxAbsError = NaN(size(exactProbability));

%% 4. Display configuration

fprintf('EXP07 Unified Formal Monte Carlo\n');

fprintf('Random Seed = %d\n',randomSeed);

fprintf('Trials per SNR = %d\n',numTrials);

fprintf('Arrival Range = [%d, %d]\n',arrivalRange);

fprintf('Common Search Range = [%d, %d]\n',searchRange);

fprintf('Buffer Length = %d\n',bufferLength);

fprintf('Total Estimates = %d\n\n', ...
    numTrials*numSNR*numChannels*numAlgorithms);

%% 5. Monte Carlo simulation

tic;

for s = 1:numSNR

    snrDb = snrList(s);

    for trial = 1:numTrials

        %% 5.1 Common random arrival delay

        D = randi(arrivalRange);

        allArrivalDelays(trial,s) = D;

        trueStart = D+Ncp+1;

        %% 5.2 Common QPSK data symbol

        dataIdx = randi([0 3],N,1);

        dataFreq = qammod(dataIdx,4, ...
            'UnitAveragePower',true);

        xData = ifft(dataFreq,N);

        dataSymbol = ...
            [xData(end-Ncp+1:end);xData];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        assert(abs(sum(abs(dataSymbol).^2) ...
            -cfg.trainingEnergy)<1e-10);

        %% 5.3 Common base noise

        baseNoise = ...
            (randn(bufferLength,1) ...
            +1j*randn(bufferLength,1))/sqrt(2);

        %% 5.4 Algorithm-specific training symbols

        trainingList = { ...
            schmidl_cox_training(cfg), ...
            park_training_original(cfg), ...
            minn_training(cfg), ...
            cazac_training(cfg)};

        %% 5.5 Process four algorithms

        for a = 1:numAlgorithms

            training = trainingList{a};

            assert(numel(training)==N+Ncp);

            assert(abs(sum(abs(training).^2) ...
                -cfg.trainingEnergy)<1e-10);

            txFrame = [training;dataSymbol];

            txBuffered = zeros(bufferLength,1);

            txBuffered(D+1:D+numel(txFrame)) = ...
                txFrame;

            %% 5.6 Process both channel conditions

            for c = 1:numChannels

                h = channelResponses{c};

                rxFull = conv(txBuffered,h);

                rxClean = rxFull(1:bufferLength);

                %% Received active-frame power

                activeFrame = ...
                    rxClean(D+1:D+numel(txFrame));

                signalPower = ...
                    mean(abs(activeFrame).^2);

                assert(isfinite(signalPower));
                assert(signalPower>0);

                allSignalPowers(trial,s,c,a) = ...
                    signalPower;

                if c==1
                    assert(abs(signalPower-1)<1e-10);
                end

                %% Add complex AWGN

                noisePower = ...
                    signalPower/10^(snrDb/10);

                rx = rxClean ...
                    +sqrt(noisePower)*baseNoise;

                %% Unified timing estimation

                [est,peak,valid,info] = ...
                    timing_estimator_unified( ...
                    rx,cfg,algorithmNames{a}, ...
                    searchRange);

                %% Verify common candidate support

                candidateStarts = ...
                    info.usefulIndex(info.searchValid);

                assert(isequal(candidateStarts, ...
                    (searchRange(1):searchRange(2)).'));

                %% Save trial results

                allValid(trial,s,c,a) = valid;

                if valid

                    assert(isfinite(est));
                    assert(est==round(est));

                    assert(est>=searchRange(1));
                    assert(est<=searchRange(2));

                    allErrors(trial,s,c,a) = ...
                        est-trueStart;

                    allPeakMetrics(trial,s,c,a) = ...
                        peak;

                end

            end

        end

    end

    fprintf('SNR = %2d dB completed.\n',snrDb);

end

elapsedTime = toc;

%% 6. Calculate statistics

for s = 1:numSNR

    for c = 1:numChannels

        for a = 1:numAlgorithms

            errors = allErrors(:,s,c,a);

            validErrors = errors(isfinite(errors));

            exactProbability(s,c,a) = ...
                mean(errors==0);

            cpSafeProbability(s,c,a) = ...
                mean(errors>=cpSafeMin(c) ...
                & errors<=0);

            failProbability(s,c,a) = ...
                mean(~allValid(:,s,c,a));

            if ~isempty(validErrors)

                conditionalMSE(s,c,a) = ...
                    mean(validErrors.^2);

                meanError(s,c,a) = ...
                    mean(validErrors);

                maxAbsError(s,c,a) = ...
                    max(abs(validErrors));

            end

        end

    end

end

%% 7. Statistical consistency checks

assert(all(isfinite(allErrors(allValid))));

assert(all(isnan(allErrors(~allValid))));

assert(all(exactProbability(:) ...
    <=cpSafeProbability(:)));

assert(all(cpSafeProbability(:) ...
    <=1-failProbability(:)+1e-12));

assert(all(isfinite(allSignalPowers(:))));

assert(all(allSignalPowers(:)>0));


%% 8. Additional statistics

correctCount = zeros(size(exactProbability));

cpSafeCount = zeros(size(exactProbability));

failCount = zeros(size(exactProbability));

errorStd = NaN(size(exactProbability));

meanSignalPower = NaN(size(exactProbability));

for s = 1:numSNR

    for c = 1:numChannels

        for a = 1:numAlgorithms

            errors = allErrors(:,s,c,a);

            validErrors = errors(isfinite(errors));

            correctCount(s,c,a) = ...
                sum(errors==0);

            cpSafeCount(s,c,a) = ...
                sum(errors>=cpSafeMin(c) ...
                & errors<=0);

            failCount(s,c,a) = ...
                sum(~allValid(:,s,c,a));

            meanSignalPower(s,c,a) = ...
                mean(allSignalPowers(:,s,c,a));

            if ~isempty(validErrors)

                errorStd(s,c,a) = ...
                    std(validErrors,1);

            end

        end

    end

end

%% 9. Additional acceptance checks

assert(numel(allErrors)== ...
    numTrials*numSNR*numChannels*numAlgorithms);

assert(all(allArrivalDelays(:)>=arrivalRange(1)));

assert(all(allArrivalDelays(:)<=arrivalRange(2)));

assert(all(abs(exactProbability(:) ...
    -correctCount(:)/numTrials)<1e-12));

assert(all(abs(cpSafeProbability(:) ...
    -cpSafeCount(:)/numTrials)<1e-12));

assert(all(abs(failProbability(:) ...
    -failCount(:)/numTrials)<1e-12));

assert(all(abs(allSignalPowers(:,:,1,:)-1) ...
    <1e-10,'all'));

% Reconstruct estimated useful-symbol starts
trueStarts = allArrivalDelays+Ncp+1;

estimatedStarts = allErrors + ...
    reshape(trueStarts,numTrials,numSNR,1,1);

validStarts = estimatedStarts(allValid);

assert(all(validStarts>=searchRange(1)));

assert(all(validStarts<=searchRange(2)));

assert(all(validStarts==round(validStarts)));

fprintf('\nAll formal experiment checks passed.\n');

%% 10. Construct summary table

[S,C,A] = ndgrid( ...
    1:numSNR, ...
    1:numChannels, ...
    1:numAlgorithms);

snrColumn = reshape(snrList(S(:)),[],1);

algorithmColumn = ...
    reshape(string(algorithmNames(A(:))),[],1);

channelColumn = ...
    reshape(string(channelNames(C(:))),[],1);

trialsColumn = ...
    repmat(numTrials,numel(S),1);

resultsTable = table( ...
    snrColumn, ...
    channelColumn, ...
    algorithmColumn, ...
    trialsColumn, ...
    exactProbability(:), ...
    cpSafeProbability(:), ...
    failProbability(:), ...
    conditionalMSE(:), ...
    meanError(:), ...
    errorStd(:), ...
    maxAbsError(:), ...
    correctCount(:), ...
    cpSafeCount(:), ...
    failCount(:), ...
    meanSignalPower(:), ...
    'VariableNames',{ ...
    'SNR_dB', ...
    'Channel', ...
    'Algorithm', ...
    'Trials', ...
    'Exact', ...
    'CPsafe', ...
    'Fail', ...
    'Conditional_MSE', ...
    'Mean_Error', ...
    'Error_Std', ...
    'Max_Abs_Error', ...
    'Correct_Count', ...
    'CPsafe_Count', ...
    'Fail_Count', ...
    'Mean_Signal_Power'});

resultsTable = sortrows(resultsTable, ...
    {'Channel','SNR_dB','Algorithm'});

%% 11. Create output directories

dataDir = fullfile(projectRoot,'results','data');

tableDir = fullfile(projectRoot,'results','tables');

figureDir = fullfile(projectRoot,'results','figures');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

if ~exist(tableDir,'dir')
    mkdir(tableDir);
end

if ~exist(figureDir,'dir')
    mkdir(figureDir);
end

%% 12. Record experiment methodology

experimentVersion = ...
    'EXP07 Unified Comparison v1.0';

searchPolicy = ...
    'Common useful-start search range [21,113]';

randomizationPolicy = ...
    ['Shared arrival delay, QPSK data and base noise; ' ...
     'algorithm-specific training symbols'];

parkVersion = ...
    ['Supervisor-confirmed mirror correlation; ' ...
     'global correlation normalization; energy gate 8'];

minnVersion = ...
    'Four-part TD Golay training [-A,+A,-A,-A]';

cazacVersion = ...
    'Length-64 ZC root 1, known 80-sample template';

scVersion = ...
    'Repeated halves, relative threshold plateau midpoint';

scThreshold = cfg.SC_threshold;

parkEnergyThreshold = 8;

%% 13. Save raw data before plotting

matPath = fullfile(dataDir, ...
    'exp07_unified_monte_carlo.mat');

save(matPath, ...
    'cfg','experimentVersion', ...
    'randomSeed','randomizationPolicy', ...
    'searchPolicy','scVersion','parkVersion', ...
    'minnVersion','cazacVersion', ...
    'scThreshold','parkEnergyThreshold', ...
    'arrivalRange','searchRange', ...
    'snrList','numTrials','bufferLength', ...
    'algorithmNames','channelNames', ...
    'channelResponses','maxPathDelays', ...
    'cpSafeMin','allArrivalDelays', ...
    'allErrors','allValid','allPeakMetrics', ...
    'allSignalPowers','estimatedStarts', ...
    'exactProbability','cpSafeProbability', ...
    'failProbability','conditionalMSE', ...
    'meanError','errorStd','maxAbsError', ...
    'correctCount','cpSafeCount','failCount', ...
    'meanSignalPower','resultsTable', ...
    'elapsedTime');

csvPath = fullfile(tableDir, ...
    'exp07_unified_monte_carlo.csv');

writetable(resultsTable,csvPath);

fprintf('\nMAT and CSV results saved successfully.\n');

%% 14. Plot configuration

displayNames = { ...
    'Schmidl-Cox','Park','Minn','CAZAC'};

channelLabels = { ...
    'AWGN','Fixed Multipath + AWGN'};

channelTags = {'awgn','multipath'};

colors = [ ...
    0.0000 0.4470 0.7410; ...
    0.8500 0.3250 0.0980; ...
    0.4660 0.6740 0.1880; ...
    0.4940 0.1840 0.5560];

markers = {'o','s','^','d'};

lineStyles = {'-','--','-.',':'};

metricData = { ...
    conditionalMSE, ...
    exactProbability, ...
    cpSafeProbability, ...
    failProbability};

metricNames = { ...
    'Timing MSE', ...
    'Exact Timing Probability', ...
    'CP-safe Probability', ...
    'Failure Probability'};

metricTags = { ...
    'mse','exact','cpsafe','fail'};

yLabels = { ...
    'Conditional Timing MSE (samples^2)', ...
    'Exact Timing Probability', ...
    'CP-safe Probability', ...
    'Failure Probability'};

%% 15. Generate eight performance figures

for c = 1:numChannels

    for k = 1:numel(metricData)

        values = metricData{k};

        fig = figure('Color','w', ...
            'Position',[100 100 1100 650]);

        hold on;

        for a = 1:numAlgorithms

            y = values(:,c,a);

            plot(snrList,y, ...
                'Color',colors(a,:), ...
                'LineStyle',lineStyles{a}, ...
                'Marker',markers{a}, ...
                'LineWidth',1.8, ...
                'MarkerSize',6, ...
                'DisplayName',displayNames{a});

        end

        grid on;

        xlabel('SNR (dB)');

        ylabel(yLabels{k});

        title(sprintf('%s: %s', ...
            metricNames{k},channelLabels{c}));

        legend('Location','best');

        xlim([min(snrList),max(snrList)]);

        xticks(snrList);

        if k~=1
            ylim([0 1.05]);
        end

        set(gca,'FontSize',12);

        figureName = sprintf( ...
            'exp07_%s_%s.png', ...
            channelTags{c},metricTags{k});

        exportgraphics(fig, ...
            fullfile(figureDir,figureName), ...
            'Resolution',300);

    end

end

%% 16. Display final results

for c = 1:numChannels

    fprintf('\n%s\n',channelLabels{c});

    fprintf('%5s %-8s %9s %9s %9s %12s %11s\n', ...
        'SNR','Method','Exact','CP-safe', ...
        'Fail','Cond.MSE','MeanError');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf(['%5d %-8s %9.4f %9.4f ' ...
                '%9.4f %12.4f %+11.4f\n'], ...
                snrList(s), ...
                algorithmNames{a}, ...
                exactProbability(s,c,a), ...
                cpSafeProbability(s,c,a), ...
                failProbability(s,c,a), ...
                conditionalMSE(s,c,a), ...
                meanError(s,c,a));

        end

    end

end

%% 17. Completion report

fprintf('\n===== EXP07 Completion Summary =====\n');

fprintf('Total Estimates = %d\n',numel(allErrors));

fprintf('Total Valid Estimates = %d\n',sum(allValid(:)));

fprintf('Total Failed Estimates = %d\n',sum(~allValid(:)));

fprintf('Summary Table Rows = %d\n',height(resultsTable));

fprintf('Simulation Time = %.2f seconds\n',elapsedTime);

fprintf('\nSaved MAT:\n%s\n',matPath);

fprintf('\nSaved CSV:\n%s\n',csvPath);

fprintf('\nSaved Figures:\n');

for c = 1:numChannels

    for k = 1:numel(metricTags)

        fprintf('%s\n',fullfile(figureDir, ...
            sprintf('exp07_%s_%s.png', ...
            channelTags{c},metricTags{k})));

    end

end

fprintf('\nEXP07 unified formal Monte Carlo completed.\n');
