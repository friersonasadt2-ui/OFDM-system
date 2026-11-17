%% EXP04 - Park Early Peak Threshold Validation
% Independent AWGN validation experiment
%
% Purpose:
% Validate the previously selected metric threshold = 0.50
% using new random samples.
%
% This is an experimental Early Peak estimator.
% It is NOT the original Park (2003) decision rule.
%
% Experimental settings:
%   FFT size            : 64
%   CP length           : 16
%   Channel             : AWGN
%   SNR                 : [0 5 10 20] dB
%   Trials per SNR      : 300
%   Arrival delay       : random integer in [20, 80]
%   Energy threshold    : 8
%   Metric threshold    : 0.50
%   Random seed         : 2030

clearvars;
clc;

%% 1. Project setup
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

rng(2030);

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

%% 2. Experiment parameters
snrList = [0 5 10 20];

thresholdList = [0.50];

numTrials = 300;

numSNR = numel(snrList);
numThresholds = numel(thresholdList);

arrivalRange = [20 80];

energyThreshold = 8;

% Result arrays: [SNR, threshold]

exact = zeros(numSNR,numThresholds);
cpSafe = zeros(numSNR,numThresholds);
fail = zeros(numSNR,numThresholds);
condMSE = NaN(numSNR,numThresholds);

fprintf('EXP04 Park Threshold Independent Validation\n');
fprintf('Trials per SNR = %d\n',numTrials);
fprintf('Random Seed = 2030\n');
fprintf('Energy Threshold = %.2f\n',energyThreshold);
fprintf('Metric Threshold = %.2f\n',thresholdList(1));

%% 3. SNR loop
for s = 1:numSNR

    snrDb = snrList(s);

    % NaN indicates no valid timing estimate.
    errors = NaN(numTrials,numThresholds);

    %% Monte Carlo loop
    for t = 1:numTrials

        % 3.1 Random arrival delay
        D = randi(arrivalRange);

        % 3.2 Original Park frequency-domain training
        pn = 2*randi([0 1],L,1)-1;

        X = zeros(N,1);
        X(1:2:end) = pn;

        x = ifft(X,N);

        training = ...
            [x(end-Ncp+1:end);x];

        training = normalize_training_energy( ...
            training,cfg.trainingEnergy);

        % 3.3 Random QPSK OFDM data symbol
        dataIdx = randi([0 3],N,1);

        dataFreq = pskmod(dataIdx,4,pi/4);

        dataTime = ifft(dataFreq,N);

        dataSymbol = ...
            [dataTime(end-Ncp+1:end);dataTime];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        % 3.4 Construct transmitted frame
        txFrame = [training;dataSymbol];

        % Leading and trailing padding
        rxClean = ...
            [zeros(D,1);txFrame;zeros(N,1)];

        % True start of the useful training symbol
        % MATLAB uses one-based indexing.
        trueStart = D+Ncp+1;

        % 3.5 Add complex AWGN
        % Signal power is measured over the active
        % transmitted frame, excluding zero padding.
        %
        % This matches the previous AWGN experiments.

        signalPower = mean(abs(txFrame).^2);

        noisePower = ...
            signalPower/(10^(snrDb/10));

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean+noise;

        % 3.6 Early Peak timing estimation
        for q = 1:numThresholds

            metricThreshold = thresholdList(q);

            [estimatedStart,~,~,isValid] = ...
                park_estimator_early_peak( ...
                rx,cfg,arrivalRange, ...
                energyThreshold,metricThreshold);

            % Record timing error only if valid.
            % Otherwise leave NaN.

            if isValid

                errors(t,q) = ...
                    estimatedStart-trueStart;

            end

        end

    end

    %% 4. Performance statistics
    for q = 1:numThresholds

        e = errors(:,q);

        % Exact probability:
        % No-estimate cases count as unsuccessful.
        exact(s,q) = mean(e==0);

        % CP-safe probability:
        % Current simplified definition [-Ncp, 0].
        % For multipath channels this condition
        % must be reconsidered using channel delay.
        cpSafe(s,q) = ...
            mean(e>=-Ncp & e<=0);

        % No-estimate probability
        fail(s,q) = mean(isnan(e));

        % Conditional timing MSE:
        % Calculated only over valid estimates.
        validErrors = e(~isnan(e));

        if ~isempty(validErrors)

            condMSE(s,q) = ...
                mean(validErrors.^2);

        end

    end

    %% 5. Print current SNR results
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

%% 6. Summary table
fprintf('Independent Validation Summary\n');
fprintf('SNR(dB)\tExact\tCP-safe\tFail\tCond.MSE\n');

for s = 1:numSNR

    fprintf('%d\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        snrList(s), ...
        exact(s,1), ...
        cpSafe(s,1), ...
        fail(s,1), ...
        condMSE(s,1));

end

%% 7. Save results
dataDir = fullfile(projectRoot,'results','data');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

save(fullfile(dataDir, ...
    'exp04_park_threshold_validation.mat'), ...
    'cfg', ...
    'snrList', ...
    'thresholdList', ...
    'numTrials', ...
    'arrivalRange', ...
    'energyThreshold', ...
    'exact', ...
    'cpSafe', ...
    'fail', ...
    'condMSE');

fprintf('\nValidation results saved.\n');
fprintf('EXP04 independent validation completed.\n');