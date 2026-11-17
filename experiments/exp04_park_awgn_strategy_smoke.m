%% EXP04 - Park AWGN Peak Strategy Smoke Test
% Paired comparison under AWGN
%
% 1. Maximum peak selection
% 2. Earliest reliable peak selection
%
% Early Peak is an experimental modification,
% NOT the original Park 2003 decision rule.

clearvars;
clc;

%% 1. Project setup
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(2028);

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

%% 2. Experiment parameters
snrList = [0 10 20];
numTrials = 100;

arrivalRange = [20 80];

energyThreshold = 8;
metricThreshold = 0.95;

fprintf('\nEXP04 Park AWGN Strategy Smoke Test\n');
fprintf('Trials per SNR = %d\n',numTrials);

%% 3. SNR loop
for s = 1:numel(snrList)

    snrDb = snrList(s);

    errorsMax = NaN(numTrials,1);
    errorsEarly = NaN(numTrials,1);

    for t = 1:numTrials

        %% Random arrival delay
        D = randi(arrivalRange);

        %% Original Park training sequence
        pn = 2*randi([0 1],L,1)-1;

        X = zeros(N,1);
        X(1:2:end) = pn;

        x = ifft(X,N);

        training = ...
            [x(end-Ncp+1:end);x];

        training = normalize_training_energy( ...
            training,cfg.trainingEnergy);

        %% QPSK OFDM data
        dataIdx = randi([0 3],N,1);
        dataFreq = pskmod(dataIdx,4,pi/4);

        dataTime = ifft(dataFreq,N);

        dataSymbol = ...
            [dataTime(end-Ncp+1:end);dataTime];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        %% Transmitted frame
        txFrame = [training;dataSymbol];

        rxClean = ...
            [zeros(D,1);txFrame;zeros(N,1)];

        trueStart = D+Ncp+1;

        %% Add complex AWGN
        % Measure signal power over the active frame.
        % Do not include the leading/trailing zero padding.
        signalPower = mean(abs(txFrame).^2);

        noisePower = ...
            signalPower / (10^(snrDb/10));

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean + noise;

        %% Maximum peak estimator
        [estMax,~,~,validMax] = ...
            park_estimator_paper2003( ...
            rx,cfg,arrivalRange,energyThreshold);

        %% Early peak estimator
        [estEarly,~,~,validEarly] = ...
            park_estimator_early_peak( ...
            rx,cfg,arrivalRange, ...
            energyThreshold,metricThreshold);

        %% Record timing errors
        if validMax
            errorsMax(t) = estMax-trueStart;
        end

        if validEarly
            errorsEarly(t) = estEarly-trueStart;
        end

    end

    %% 4. Calculate statistics
    exactMax = mean(errorsMax==0);
    exactEarly = mean(errorsEarly==0);

    safeMax = mean(errorsMax>=-Ncp & errorsMax<=0);
    safeEarly = mean(errorsEarly>=-Ncp & errorsEarly<=0);

    failMax = mean(isnan(errorsMax));
    failEarly = mean(isnan(errorsEarly));

    validMaxErrors = errorsMax(~isnan(errorsMax));
    validEarlyErrors = errorsEarly(~isnan(errorsEarly));

    mseMax = NaN;
    mseEarly = NaN;

    if ~isempty(validMaxErrors)
        mseMax = mean(validMaxErrors.^2);
    end

    if ~isempty(validEarlyErrors)
        mseEarly = mean(validEarlyErrors.^2);
    end

    %% 5. Print results
    fprintf('\nSNR = %d dB\n',snrDb);
    fprintf('Method\t\tExact\tCP-safe\tFail\tCond.MSE\n');

    fprintf('Max Peak\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        exactMax,safeMax,failMax,mseMax);

    fprintf('Early Peak\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        exactEarly,safeEarly,failEarly,mseEarly);

end

fprintf('\nAWGN smoke test completed.\n');