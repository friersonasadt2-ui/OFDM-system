%% EXP04 - Park Early Peak Threshold Sensitivity
% Exploratory AWGN experiment, not final performance evaluation.
%
% All thresholds are evaluated on the same received
% signal within each trial.
%
% Early Peak is an experimental modification,
% NOT the original Park (2003) decision rule.

clearvars;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(2029);

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

snrList = [0 10 20];
thresholdList = [0.30 0.50 0.70 0.95];

numTrials = 100;
numThresholds = numel(thresholdList);
numSNR = numel(snrList);

arrivalRange = [20 80];
energyThreshold = 8;

% Dimensions: [SNR, threshold]
exact = zeros(numSNR,numThresholds);
cpSafe = zeros(numSNR,numThresholds);
fail = zeros(numSNR,numThresholds);
condMSE = NaN(numSNR,numThresholds);

fprintf('\nEXP04 Park Threshold Sensitivity\n');
fprintf('Trials per SNR = %d\n',numTrials);

%% SNR loop

for s = 1:numSNR

    snrDb = snrList(s);

    errors = NaN(numTrials,numThresholds);

    for t = 1:numTrials

        %% Random arrival delay
        D = randi(arrivalRange);

        %% Original Park training construction
        pn = 2*randi([0 1],L,1)-1;

        X = zeros(N,1);
        X(1:2:end) = pn;

        x = ifft(X,N);

        training = [x(end-Ncp+1:end);x];

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

        %% AWGN
        signalPower = mean(abs(txFrame).^2);

        noisePower = ...
            signalPower/10^(snrDb/10);

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean+noise;

        %% Test all thresholds on SAME received signal
        for q = 1:numThresholds

            threshold = thresholdList(q);

            [estimatedStart,~,~,isValid] = ...
                park_estimator_early_peak( ...
                rx,cfg,arrivalRange, ...
                energyThreshold,threshold);

            if isValid
                errors(t,q) = ...
                    estimatedStart-trueStart;
            end

        end

    end

    %% Statistics
    for q = 1:numThresholds

        e = errors(:,q);

        exact(s,q) = mean(e==0);

        cpSafe(s,q) = ...
            mean(e>=-Ncp & e<=0);

        fail(s,q) = mean(isnan(e));

        validErrors = e(~isnan(e));

        if ~isempty(validErrors)
            condMSE(s,q) = mean(validErrors.^2);
        end

    end

    %% Print results for current SNR
    fprintf('\nSNR = %d dB\n',snrDb);

    fprintf('Threshold\tExact\tCP-safe\tFail\tCond.MSE\n');

    for q = 1:numThresholds

        fprintf('%.2f\t\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
            thresholdList(q), ...
            exact(s,q), ...
            cpSafe(s,q), ...
            fail(s,q), ...
            condMSE(s,q));

    end

end

%% Save exploratory results
dataDir = fullfile(projectRoot,'results','data');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

save(fullfile(dataDir, ...
    'exp04_park_threshold_sensitivity.mat'), ...
    'cfg','snrList','thresholdList','numTrials', ...
    'arrivalRange','energyThreshold', ...
    'exact','cpSafe','fail','condMSE');

fprintf('\nThreshold sensitivity test completed.\n');