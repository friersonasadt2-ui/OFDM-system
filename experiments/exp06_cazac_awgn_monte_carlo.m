%% EXP06 - CAZAC AWGN Monte Carlo
% Zadoff-Chu root u=1
% Global normalized matched-correlation estimator
clear; clc;

%% 1. Initialize
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(2034);

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

%% 2. Statistics
exactProb = zeros(numSNR,1);
cpSafeProb = zeros(numSNR,1);
failProb = zeros(numSNR,1);

condMSE = NaN(numSNR,1);
meanError = NaN(numSNR,1);
errorStd = NaN(numSNR,1);

allErrors = NaN(numTrials,numSNR);
allDelays = zeros(numTrials,numSNR);

%% 3. CAZAC training
[training,~,~] = cazac_training(cfg);

assert(numel(training)==N+Ncp);
assert(abs(sum(abs(training).^2)-cfg.trainingEnergy)<1e-10);

fprintf('\nEXP06 CAZAC AWGN Monte Carlo\n');
fprintf('Nfft=%d, Ncp=%d\n',N,Ncp);
fprintf('Trials per SNR=%d\n',numTrials);
fprintf('Random Seed=2034\n');
fprintf('Training Energy=%.6f\n',sum(abs(training).^2));

%% 4. Monte Carlo
tic;

for s = 1:numSNR

    snrDb = snrList(s);
    errors = NaN(numTrials,1);

    for t = 1:numTrials

        % A. Random QPSK OFDM data
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

        % E. CAZAC timing estimation
        [estimatedStart,~,isValid,peakIndex] = ...
            cazac_estimator(rx,cfg);

        if isValid && isfinite(estimatedStart)

            assert(estimatedStart==peakIndex+Ncp, ...
                'Training-to-FFT index mismatch.');

            errors(t) = estimatedStart-trueStart;

        end

    end

    %% Statistics for current SNR
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
    'exp06_cazac_awgn_monte_carlo.mat');

csvPath = fullfile(tableDir, ...
    'exp06_cazac_awgn_monte_carlo.csv');

save(matPath, ...
    'cfg','snrList','numTrials','allErrors', ...
    'allDelays','condMSE','exactProb', ...
    'cpSafeProb','failProb','meanError', ...
    'errorStd','resultsTable','elapsedTime');

writetable(resultsTable,csvPath);

%% 7. Timing MSE
fig1 = figure('Color','w');

plot(snrList,condMSE,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('Timing MSE (samples^2)');
title('CAZAC Timing MSE under AWGN');

exportgraphics(fig1, ...
    fullfile(figureDir,'exp06_cazac_awgn_mse.png'), ...
    'Resolution',300);

%% 8. Exact Timing Probability
fig2 = figure('Color','w');

plot(snrList,exactProb,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('Exact Timing Probability');
title('CAZAC Exact Timing Probability under AWGN');
ylim([0 1.05]);

exportgraphics(fig2, ...
    fullfile(figureDir,'exp06_cazac_awgn_exact.png'), ...
    'Resolution',300);

%% 9. CP-safe Timing Probability
fig3 = figure('Color','w');

plot(snrList,cpSafeProb,'-o', ...
    'LineWidth',1.8,'MarkerSize',6);

grid on;
xlabel('SNR (dB)');
ylabel('CP-safe Timing Probability');
title('CAZAC CP-safe Timing Probability under AWGN');
ylim([0 1.05]);

exportgraphics(fig3, ...
    fullfile(figureDir,'exp06_cazac_awgn_cpsafe.png'), ...
    'Resolution',300);

%% 10. Verify saved files
assert(isfile(matPath));
assert(isfile(csvPath));

fprintf('\nAll AWGN result files saved.\n');
fprintf('EXP06 CAZAC AWGN Monte Carlo completed.\n');
