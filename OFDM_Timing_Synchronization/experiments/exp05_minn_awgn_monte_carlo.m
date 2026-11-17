%% EXP05 - Minn AWGN Monte Carlo
% Minn TD Training, Global Maximum Metric
% Nfft = 64, Ncp = 16
% 10000 trials per SNR

clear; clc;

%% 1. Initialize project
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(2031);

N = cfg.Nfft;
Ncp = cfg.Ncp;

snrList = cfg.SNR_dB;
numTrials = cfg.numTrials;
numSNR = numel(snrList);

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

%% 3. Generate fixed Minn training symbol
[training,~,~] = minn_training(cfg);

assert(numel(training)==N+Ncp);
assert(abs(sum(abs(training).^2)-cfg.trainingEnergy)<1e-10);

fprintf('\nEXP05 Minn AWGN Monte Carlo\n');
fprintf('Nfft = %d, Ncp = %d\n',N,Ncp);
fprintf('Trials per SNR = %d\n',numTrials);
fprintf('Random Seed = 2031\n');
fprintf('Training Energy = %.6f\n',sum(abs(training).^2));

%% 4. Monte Carlo simulation
tic;

for s = 1:numSNR

    snrDb = snrList(s);

    errors = NaN(numTrials,1);
    validFlags = false(numTrials,1);

    for t = 1:numTrials

        % A. Random QPSK data
        dataIndices = randi([0 3],N,1);
        dataQPSK = pskmod(dataIndices,4,pi/4);

        dataTime = ifft(dataQPSK,N);

        dataSymbol = ...
            [dataTime(end-Ncp+1:end);dataTime];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        % B. Random arrival delay
        D = randi([20 80]);

        trueStart = D+Ncp+1;
        allDelays(t,s) = D;

        % C. Transmitted frame
        txFrame = [training;dataSymbol];

        rxClean = ...
            [zeros(D,1);txFrame;zeros(N,1)];

        % D. Received active-frame SNR
        signalPower = mean(abs(txFrame).^2);

        noisePower = signalPower/10^(snrDb/10);

        noise = sqrt(noisePower/2)* ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean+noise;

        % E. Minn timing estimation
        [estimatedStart,~,isValid,~] = ...
            minn_estimator(rx,cfg);

        validFlags(t) = isValid;

        if isValid && isfinite(estimatedStart)
            errors(t) = estimatedStart-trueStart;
        end

    end

    %% F. Statistics at this SNR
    allErrors(:,s) = errors;

    validErrors = errors(isfinite(errors));

    exactProb(s) = mean(errors==0);

    cpSafeProb(s) = ...
        mean(errors>=-Ncp & errors<=0);

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

fprintf('\nMonte Carlo completed.\n');
fprintf('Elapsed Time = %.2f seconds\n',elapsedTime);

%% 5. Create results table
resultsTable = table( ...
    snrList(:),condMSE,exactProb,cpSafeProb, ...
    failProb,meanError,errorStd, ...
    'VariableNames', ...
    {'SNR_dB','TimingMSE','ExactProbability', ...
    'CPsafeProbability','FailProbability', ...
    'MeanError','ErrorStd'});

disp(resultsTable);

%% 6. Save numerical results
matPath = fullfile(dataDir, ...
    'exp05_minn_awgn_monte_carlo.mat');

csvPath = fullfile(tableDir, ...
    'exp05_minn_awgn_monte_carlo.csv');

save(matPath, ...
    'cfg','snrList','numTrials','allErrors', ...
    'allDelays','condMSE','exactProb', ...
    'cpSafeProb','failProb','meanError', ...
    'errorStd','resultsTable','elapsedTime');

writetable(resultsTable,csvPath);

%% 7. Plot Timing MSE
fig1 = figure('Color','w');

plot(snrList,condMSE,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('Timing MSE (samples^2)');
title('Minn Timing MSE under AWGN');

exportgraphics(fig1, ...
    fullfile(figureDir,'exp05_minn_awgn_mse.png'), ...
    'Resolution',300);

%% 8. Plot Exact Timing Probability
fig2 = figure('Color','w');

plot(snrList,exactProb,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('Exact Timing Probability');
title('Minn Exact Timing Probability under AWGN');
ylim([0 1.05]);

exportgraphics(fig2, ...
    fullfile(figureDir,'exp05_minn_awgn_exact.png'), ...
    'Resolution',300);

%% 9. Plot CP-safe Timing Probability
fig3 = figure('Color','w');

plot(snrList,cpSafeProb,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('CP-safe Timing Probability');
title('Minn CP-safe Timing Probability under AWGN');
ylim([0 1.05]);

exportgraphics(fig3, ...
    fullfile(figureDir,'exp05_minn_awgn_cpsafe.png'), ...
    'Resolution',300);

%% 10. Verify saved outputs
assert(isfile(matPath),'MAT file was not saved.');
assert(isfile(csvPath),'CSV file was not saved.');

fprintf('\nAll result files saved successfully.\n');
fprintf('EXP05 Minn AWGN Monte Carlo completed.\n');