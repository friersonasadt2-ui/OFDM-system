%% EXP05 - Minn Fixed Multipath Monte Carlo
clear; clc;

%% 1. Initialize
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(2033);

N = cfg.Nfft;
Ncp = cfg.Ncp;

snrList = cfg.SNR_dB;
numTrials = cfg.numTrials;
numSNR = numel(snrList);

h = cfg.hMultipath(:);
maxDelay = numel(h)-1;

assert(numel(h)==3);
assert(maxDelay<Ncp);
assert(abs(sum(abs(h).^2)-1)<1e-10);

dataDir = fullfile(projectRoot,'results','data');
tableDir = fullfile(projectRoot,'results','tables');
figureDir = fullfile(projectRoot,'results','figures');

if ~exist(dataDir,'dir'); mkdir(dataDir); end
if ~exist(tableDir,'dir'); mkdir(tableDir); end
if ~exist(figureDir,'dir'); mkdir(figureDir); end

%% 2. Initialize statistics
exactProb = zeros(numSNR,1);
cpSafeProb = zeros(numSNR,1);
failProb = zeros(numSNR,1);

condMSE = NaN(numSNR,1);
meanError = NaN(numSNR,1);
errorStd = NaN(numSNR,1);

allErrors = NaN(numTrials,numSNR);
allDelays = zeros(numTrials,numSNR);

%% 3. Minn training
[training,~,~] = minn_training(cfg);

fprintf('\nEXP05 Minn Fixed Multipath Monte Carlo\n');
fprintf('Nfft=%d, Ncp=%d\n',N,Ncp);
fprintf('Trials per SNR=%d\n',numTrials);
fprintf('Random Seed=2033\n');
fprintf('Channel Taps: ');
fprintf('%.4f ',h);
fprintf('\n');
fprintf('CP-safe error range=[%d,0]\n',-(Ncp-maxDelay));

%% 4. Monte Carlo
tic;

for s = 1:numSNR

    snrDb = snrList(s);

    errors = NaN(numTrials,1);

    for t = 1:numTrials

        % Random QPSK data
        dataIndices = randi([0 3],N,1);
        dataQPSK = pskmod(dataIndices,4,pi/4);

        dataTime = ifft(dataQPSK,N);
        dataSymbol = ...
            [dataTime(end-Ncp+1:end);dataTime];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        % Random arrival delay
        D = randi([20 80]);
        trueStart = D+Ncp+1;
        allDelays(t,s) = D;

        % Transmit buffer
        txFrame = [training;dataSymbol];
        txBuffer = [zeros(D,1);txFrame;zeros(N,1)];

        % Fixed multipath channel
        rxClean = conv(txBuffer,h);

        % Received active-frame power
        activeStart = D+1;
        activeEnd = D+numel(txFrame);

        signalPower = mean( ...
            abs(rxClean(activeStart:activeEnd)).^2);

        % Add AWGN after multipath
        noisePower = signalPower/10^(snrDb/10);

        noise = sqrt(noisePower/2)* ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean+noise;

        % Timing estimation
        [estimatedStart,~,isValid,~] = ...
            minn_estimator(rx,cfg);

        if isValid && isfinite(estimatedStart)
            errors(t) = estimatedStart-trueStart;
        end

    end

    %% Statistics
    allErrors(:,s) = errors;
    validErrors = errors(isfinite(errors));

    exactProb(s) = mean(errors==0);

    cpSafeProb(s) = mean( ...
        errors>=-(Ncp-maxDelay) & errors<=0);

    failProb(s) = mean(~isfinite(errors));

    if ~isempty(validErrors)
        condMSE(s) = mean(validErrors.^2);
        meanError(s) = mean(validErrors);
        errorStd(s) = std(validErrors,1);
    end

    fprintf(['SNR=%2d dB | Exact=%.4f | ' ...
        'CP-safe=%.4f | Fail=%.4f | MSE=%.4f\n'], ...
        snrDb,exactProb(s),cpSafeProb(s), ...
        failProb(s),condMSE(s));

end

elapsedTime = toc;

fprintf('\nCompleted in %.2f seconds.\n',elapsedTime);

%% 5. Results table
resultsTable = table( ...
    snrList(:),condMSE,exactProb,cpSafeProb, ...
    failProb,meanError,errorStd, ...
    'VariableNames', ...
    {'SNR_dB','TimingMSE','ExactProbability', ...
    'CPsafeProbability','FailProbability', ...
    'MeanError','ErrorStd'});

disp(resultsTable);

%% 6. Save MAT and CSV
matPath = fullfile(dataDir, ...
    'exp05_minn_multipath_monte_carlo.mat');

csvPath = fullfile(tableDir, ...
    'exp05_minn_multipath_monte_carlo.csv');

save(matPath, ...
    'cfg','h','maxDelay','snrList','numTrials', ...
    'allErrors','allDelays','condMSE', ...
    'exactProb','cpSafeProb','failProb', ...
    'meanError','errorStd','resultsTable', ...
    'elapsedTime');

writetable(resultsTable,csvPath);

%% 7. Plot Timing MSE
fig1 = figure('Color','w');

plot(snrList,condMSE,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('Timing MSE (samples^2)');
title('Minn Timing MSE - Fixed Multipath');

exportgraphics(fig1, ...
    fullfile(figureDir, ...
    'exp05_minn_multipath_mse.png'), ...
    'Resolution',300);

%% 8. Plot Exact Probability
fig2 = figure('Color','w');

plot(snrList,exactProb,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('Exact Timing Probability');
title('Minn Exact Timing - Fixed Multipath');
ylim([0 1.05]);

exportgraphics(fig2, ...
    fullfile(figureDir, ...
    'exp05_minn_multipath_exact.png'), ...
    'Resolution',300);

%% 9. Plot CP-safe Probability
fig3 = figure('Color','w');

plot(snrList,cpSafeProb,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('CP-safe Timing Probability');
title('Minn CP-safe Timing - Fixed Multipath');
ylim([0 1.05]);

exportgraphics(fig3, ...
    fullfile(figureDir, ...
    'exp05_minn_multipath_cpsafe.png'), ...
    'Resolution',300);

%% 10. Verify saved outputs
assert(isfile(matPath));
assert(isfile(csvPath));

fprintf('\nAll multipath result files saved.\n');
fprintf('EXP05 Minn Multipath Monte Carlo completed.\n');
