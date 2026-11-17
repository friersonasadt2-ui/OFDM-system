
%% EXP04 - Modified Park-like AWGN Monte Carlo
% Diagnostic baseline, NOT original Park (2003).

clearvars;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(cfg.randomSeed);

N = cfg.Nfft;
Ncp = cfg.Ncp;
D = cfg.arrivalDelay;

% Initial smoke-test settings
testSNR = [0 10 20];
numTrials = 100;

trueStart = D + Ncp + 1;

timingErrors = zeros(numTrials,numel(testSNR));
peakMetrics = zeros(numTrials,numel(testSNR));

for s = 1:numel(testSNR)

    SNRdB = testSNR(s);

    for t = 1:numTrials

        % Generate training
        training = park_like_training(cfg);

        % Generate data OFDM symbol
        dataIdx = randi([0 3],N,1);
        dataFreq = pskmod(dataIdx,4,pi/4);
        dataTime = ifft(dataFreq,N);

        dataSymbol = [dataTime(end-Ncp+1:end);dataTime];
        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        txFrame = [training;dataSymbol];

        % Known arrival delay
        rxClean = [zeros(D,1);txFrame;zeros(N,1)];

        % AWGN based on active-frame power
        signalPower = mean(abs(txFrame).^2);
        noisePower = signalPower/10^(SNRdB/10);

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rxNoisy = rxClean + noise;

        % Timing metric and estimation
        [~,M,centers] = park_like_metric(rxNoisy,N);

        [peakMetrics(t,s),idx] = max(M);

        estimatedStart = centers(idx)-N/2;

        timingErrors(t,s) = estimatedStart-trueStart;
    end
end

% Calculate statistical metrics
timingMSE = mean(timingErrors.^2,1);
exactProb = mean(timingErrors==0,1);
cpSafeProb = mean(timingErrors>=-Ncp & ...
                  timingErrors<=0,1);

% Results table
resultsTable = table( ...
    testSNR(:), ...
    timingMSE(:), ...
    exactProb(:), ...
    cpSafeProb(:), ...
    'VariableNames', ...
    {'SNR_dB','TimingMSE','ExactProbability', ...
     'CPSafeProbability'});

disp(resultsTable);

% Save results
dataDir = fullfile(projectRoot,'results','data');
tableDir = fullfile(projectRoot,'results','tables');

if ~exist(dataDir,'dir'); mkdir(dataDir); end
if ~exist(tableDir,'dir'); mkdir(tableDir); end

save(fullfile(dataDir, ...
    'exp04_park_like_awgn_smoke.mat'), ...
    'cfg','testSNR','numTrials','timingErrors', ...
    'peakMetrics','resultsTable');

writetable(resultsTable,fullfile(tableDir, ...
    'exp04_park_like_awgn_smoke.csv'));

fprintf('\nPark-like Monte Carlo completed and saved.\n');

% Error diagnostics
for s = 1:numel(testSNR)
    e = timingErrors(:,s);

    fprintf('\nSNR = %d dB\n',testSNR(s));
    fprintf('Nonzero errors: %d\n',nnz(e));
    fprintf('Maximum absolute error: %d\n',max(abs(e)));
    fprintf('Mean error: %.4f\n',mean(e));
end
