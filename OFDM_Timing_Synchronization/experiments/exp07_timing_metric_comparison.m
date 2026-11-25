%% EXP07 Unified Timing Metric Comparison
% Four timing synchronization algorithms
% Noiseless, AWGN and fixed multipath conditions

clearvars; clc; close all;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2043;
rng(randomSeed);

%% 1. Common configuration

N = cfg.Nfft;
Ncp = cfg.Ncp;

D = cfg.arrivalDelay;

arrivalRange = [20 80];

trueStartRange = arrivalRange+Ncp+1;

searchRange = [ ...
    trueStartRange(1)-Ncp, ...
    trueStartRange(2)+Ncp];

trueStart = D+Ncp+1;

snrDb = 10;

algorithmNames = {'SC','Park','Minn','CAZAC'};

displayNames = {'Schmidl-Cox','Park','Minn','CAZAC'};

scenarioNames = { ...
    'Noiseless', ...
    'AWGN', ...
    'Fixed_Multipath'};

numAlgorithms = numel(algorithmNames);
numScenarios = numel(scenarioNames);

h = cfg.hMultipath(:);

maxPathDelay = numel(h)-1;

bufferLength = arrivalRange(2) ...
    +2*(N+Ncp)+N+maxPathDelay;

%% 2. Generate common QPSK data

dataIdx = randi([0 3],N,1);

dataFreq = qammod(dataIdx,4, ...
    'UnitAveragePower',true);

xData = ifft(dataFreq,N);

dataSymbol = [xData(end-Ncp+1:end);xData];

dataSymbol = normalize_training_energy( ...
    dataSymbol,cfg.trainingEnergy);

%% 3. Generate four training symbols

trainingList = { ...
    schmidl_cox_training(cfg), ...
    park_training_original(cfg), ...
    minn_training(cfg), ...
    cazac_training(cfg)};

for a = 1:numAlgorithms

    assert(numel(trainingList{a})==N+Ncp);

    assert(abs(sum(abs(trainingList{a}).^2) ...
        -cfg.trainingEnergy)<1e-10);

end

%% 4. Generate common base noise

baseNoise = ...
    (randn(bufferLength,1) ...
    +1j*randn(bufferLength,1))/sqrt(2);

%% 5. Initialize results

rawMetric = cell(numScenarios,numAlgorithms);

plotMetric = cell(numScenarios,numAlgorithms);

candidateErrors = cell(numScenarios,numAlgorithms);

validCandidateMasks = cell(numScenarios,numAlgorithms);

estimatedStarts = NaN(numScenarios,numAlgorithms);

timingErrors = NaN(numScenarios,numAlgorithms);

peakMetrics = NaN(numScenarios,numAlgorithms);

isValid = false(numScenarios,numAlgorithms);

signalPowers = NaN(numScenarios,numAlgorithms);

%% 6. Calculate timing metrics

fprintf('EXP07 Timing Metric Comparison\n');

fprintf('Random Seed = %d\n',randomSeed);

fprintf('True FFT Start = %d\n',trueStart);

fprintf('Common Search Range = [%d, %d]\n',searchRange);

fprintf('Noisy Scenario SNR = %.2f dB\n\n',snrDb);

for s = 1:numScenarios

    fprintf('\n%s\n',scenarioNames{s});

    for a = 1:numAlgorithms

        training = trainingList{a};

        txFrame = [training;dataSymbol];

        txBuffered = zeros(bufferLength,1);

        txBuffered(D+1:D+numel(txFrame)) = txFrame;

        %% Channel processing

        if s==3

            rxFull = conv(txBuffered,h);

            % Preserve the common receiver buffer length
            rxClean = rxFull(1:bufferLength);

        else

            rxClean = txBuffered;

        end

        %% Received active-frame signal power

        activeFrame = ...
            rxClean(D+1:D+numel(txFrame));

        signalPower = mean(abs(activeFrame).^2);

        assert(isfinite(signalPower) ...
            && signalPower>0);

        signalPowers(s,a) = signalPower;

        %% Add noise only in noisy scenarios

        if s==1

            rx = rxClean;

        else

            noisePower = ...
                signalPower/10^(snrDb/10);

            rx = rxClean ...
                +sqrt(noisePower)*baseNoise;

        end

        %% Unified timing estimation

        [est,peak,valid,info] = ...
            timing_estimator_unified( ...
            rx,cfg,algorithmNames{a},searchRange);

        estimatedStarts(s,a) = est;

        peakMetrics(s,a) = peak;

        isValid(s,a) = valid;

        if valid
            timingErrors(s,a) = est-trueStart;
        end

        %% Extract common search region

        mask = info.searchValid;

        d = info.usefulIndex(mask);

        m = info.metric(mask);

        assert(isequal(d, ...
            (searchRange(1):searchRange(2)).'));

        assert(all(isfinite(m)));

        rawMetric{s,a} = m;

        candidateErrors{s,a} = d-trueStart;

        validCandidateMasks{s,a} = ...
            info.validCandidates(mask);

        %% Visualization-only normalization

        maxValue = max(m);

        if maxValue>0

            plotMetric{s,a} = m/maxValue;

        else

            plotMetric{s,a} = zeros(size(m));

        end

        %% Print results

        fprintf('%-8s | Estimate=%6.0f | Error=%+5.0f | Valid=%d\n', ...
            algorithmNames{a},est, ...
            timingErrors(s,a),valid);

    end

end

%% 7. Create output directories

figureDir = fullfile(projectRoot,'results','figures');

dataDir = fullfile(projectRoot,'results','data');

if ~exist(figureDir,'dir')
    mkdir(figureDir);
end

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

%% 8. Plot comparison figures

plotTitles = { ...
    'Timing Metric Comparison: Noiseless', ...
    'Timing Metric Comparison: 10 dB AWGN', ...
    'Timing Metric Comparison: 10 dB Fixed Multipath + AWGN'};

figureNames = { ...
    'exp07_metric_noiseless.png', ...
    'exp07_metric_awgn_10dB.png', ...
    'exp07_metric_multipath_10dB.png'};

colors = [ ...
    0.0000 0.4470 0.7410; ...
    0.8500 0.3250 0.0980; ...
    0.4660 0.6740 0.1880; ...
    0.4940 0.1840 0.5560];

lineStyles = {'-','--','-.',':'};

for s = 1:numScenarios

    fig = figure('Color','w', ...
        'Position',[100 100 1100 650]);

    hold on;

    for a = 1:numAlgorithms

        plot(candidateErrors{s,a}, ...
            plotMetric{s,a}, ...
            'Color',colors(a,:), ...
            'LineStyle',lineStyles{a}, ...
            'LineWidth',1.8, ...
            'DisplayName',displayNames{a});

        % Mark the actual selected estimate
        if isValid(s,a)

            e = timingErrors(s,a);

            idx = find(candidateErrors{s,a}==e,1);

            if ~isempty(idx)

                plot(e,plotMetric{s,a}(idx), ...
                    'o', ...
                    'Color',colors(a,:), ...
                    'MarkerFaceColor',colors(a,:), ...
                    'MarkerSize',6, ...
                    'HandleVisibility','off');

            end

        end

    end

    xline(0,'k--','True FFT Start', ...
        'LineWidth',1.2, ...
        'HandleVisibility','off');

    grid on;

    xlabel('Candidate Timing Error (samples)');

    ylabel('Timing Metric (Display-normalized)');

    title(plotTitles{s});

    legend('Location','best');

    xlim(searchRange-trueStart);

    ylim([0 1.05]);

    set(gca,'FontSize',12);

    exportgraphics(fig, ...
        fullfile(figureDir,figureNames{s}), ...
        'Resolution',300);

end

%% 9. Save all metric data

save(fullfile(dataDir, ...
    'exp07_timing_metric_comparison.mat'), ...
    'cfg','randomSeed','snrDb', ...
    'algorithmNames','scenarioNames', ...
    'arrivalRange','searchRange', ...
    'D','trueStart','bufferLength', ...
    'trainingList','dataSymbol','baseNoise', ...
    'rawMetric','plotMetric', ...
    'candidateErrors','validCandidateMasks', ...
    'estimatedStarts','timingErrors', ...
    'peakMetrics','isValid','signalPowers');

%% 10. Display summary

fprintf('\nTiming Error Summary\n');

errorTable = array2table(timingErrors, ...
    'VariableNames',algorithmNames, ...
    'RowNames',scenarioNames);

disp(errorTable);

fprintf('Raw metrics and display metrics saved.\n');

fprintf('Three comparison figures saved.\n');

fprintf('\nEXP07 timing metric comparison completed.\n');