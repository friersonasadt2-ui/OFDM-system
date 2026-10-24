%% Experiment 03E-6: Schmidl-Cox Monte Carlo Test

clear;
clc;
close all;

%% 1. Unified Experiment Configuration
cfg = default_config();

% OFDM parameters
Nfft = cfg.Nfft;
Ncp = cfg.Ncp;
L = Nfft / 2;

% Timing parameters
D = cfg.arrivalDelay;
alpha = cfg.SC_threshold;

% Monte Carlo parameters
SNR_dB = cfg.SNR_dB;
numTrials = cfg.numTrials;

% Reproducibility
rng(cfg.randomSeed);

%% 2. Generate Schmidl-Cox Training Symbol
qpskData = randi([0 3], L, 1);

qpskSymbols = pskmod(qpskData, 4, pi/4);

X = zeros(Nfft, 1);

X(1:2:end) = qpskSymbols;

x = ifft(X, Nfft);

cp = x(end-Ncp+1:end);

trainingSymbol = [cp; x];

%% 3. Generate One Data OFDM Symbol
dataIndices = randi([0 3], Nfft, 1);

dataSymbols = pskmod(dataIndices, 4, pi/4);

dataTime = ifft(dataSymbols, Nfft);

dataCP = dataTime(end-Ncp+1:end);

dataOFDM = [dataCP; dataTime];

%% Normalize Data OFDM Symbol Energy
% Match preamble and data average power (both have 80 samples).
dataOFDM = normalize_training_energy(dataOFDM, cfg.trainingEnergy);

%% Normalize Training Symbol Energy

trainingSymbol = normalize_training_energy( ...
    trainingSymbol, cfg.trainingEnergy);

%% 4. Construct Transmitted Frame
txFrame = [trainingSymbol; dataOFDM];

txDelayed = [zeros(D,1); txFrame];

trueUsefulStart = D + Ncp + 1;

%% 5. AWGN Parameters

% Calculate clean received signal power over the active frame.
% Exclude the leading zero-padding used for timing delay.
signalPower = mean(abs(txDelayed(D+1:D+numel(txFrame))).^2);

assert(isfinite(signalPower) && signalPower > 0, ...
    'Invalid signal power.');

fprintf('AWGN clean signal power = %.6f\n', signalPower);
fprintf('Training energy = %.6f\n', sum(abs(trainingSymbol).^2));

%% 6. Initialize Monte Carlo Results
numSNR = length(SNR_dB);

timingMSE = zeros(1, numSNR);

exactProbability = zeros(1, numSNR);

cpSafeProbability = zeros(1, numSNR);

detectionFailureRate = zeros(1, numSNR);

% Store timing errors for selected SNR values
selectedSNR = [0, 10, 20];

timingErrorSamples = NaN(numTrials, length(selectedSNR));

%% 7. Monte Carlo Simulation
simulationTimer = tic;

for snrIndex = 1:numSNR

    currentSNR = SNR_dB(snrIndex);

    % Noise power for current SNR
    noisePowerCurrent = signalPower / ...
        (10^(currentSNR/10));

    % Initialize trial results
    timingErrors = NaN(numTrials, 1);

    exactSuccess = false(numTrials, 1);

    cpSafeSuccess = false(numTrials, 1);

    detectionSuccess = false(numTrials, 1);

    %% Monte Carlo Trials
    for trial = 1:numTrials

        % Generate independent complex AWGN
        noise = sqrt(noisePowerCurrent/2) * ...
            (randn(size(txDelayed)) + ...
             1j*randn(size(txDelayed)));

        % Received signal
        rxNoisy = txDelayed + noise;

        % Schmidl-Cox timing metric
        [~, ~, M] = schmidl_cox_metric(rxNoisy, L);

        % Timing estimation
        [estimatedStart, info] = ...
            schmidl_cox_estimator(M, alpha);

        % Record result
        if info.detected && isfinite(estimatedStart)

            detectionSuccess(trial) = true;

            timingErrors(trial) = ...
                estimatedStart - trueUsefulStart;

            % Save timing errors for selected SNR values
            selectedIndex = find(selectedSNR == currentSNR, 1);

            if ~isempty(selectedIndex)
                timingErrorSamples(trial, selectedIndex) = timingErrors(trial);
            end

            exactSuccess(trial) = ...
                (timingErrors(trial) == 0);

            cpSafeSuccess(trial) = ...
                (timingErrors(trial) >= -Ncp) && ...
                (timingErrors(trial) <= 0);

        end

    end  % End of Monte Carlo trials

    %% Calculate statistics for current SNR
    validTrials = detectionSuccess;

    if any(validTrials)
        timingMSE(snrIndex) = ...
            mean(timingErrors(validTrials).^2);
    else
        timingMSE(snrIndex) = NaN;
    end

    exactProbability(snrIndex) = mean(exactSuccess);

    cpSafeProbability(snrIndex) = mean(cpSafeSuccess);

    detectionFailureRate(snrIndex) = mean(~detectionSuccess);

    %% Display current SNR results

    fprintf(['SNR = %2d dB | MSE = %8.3f | ', ...
             'Exact = %.3f | CP-safe = %.3f | ', ...
             'Failure = %.3f\n'], ...
             currentSNR, ...
             timingMSE(snrIndex), ...
             exactProbability(snrIndex), ...
             cpSafeProbability(snrIndex), ...
             detectionFailureRate(snrIndex));

end  % End of SNR sweep

%% 8. Simulation Runtime
elapsedTime = toc(simulationTimer);

fprintf('\n--- Simulation Runtime ---\n');

fprintf('Total trials: %d\n', numTrials * numSNR);

fprintf('Elapsed time: %.2f seconds\n', elapsedTime);

fprintf('Average time per trial: %.6f seconds\n', ...
    elapsedTime / (numTrials * numSNR));

disp('Monte Carlo SNR sweep completed.');

%% 9. Save Monte Carlo Results
% Get centralized project paths
paths = project_paths();

% Create results table
resultsTable = table( ...
    SNR_dB(:), ...
    timingMSE(:), ...
    exactProbability(:), ...
    cpSafeProbability(:), ...
    detectionFailureRate(:), ...
    'VariableNames', { ...
    'SNR_dB', ...
    'Timing_MSE', ...
    'Exact_Probability', ...
    'CP_Safe_Probability', ...
    'No_Estimate_Rate'});

% Define output filenames
fileBase = sprintf('sc_awgn_monte_carlo_%d', numTrials);

% Save CSV to results/tables
csvPath = fullfile(paths.tables, [fileBase, '.csv']);

writetable(resultsTable, csvPath);

% Save MAT to results/data
matPath = fullfile(paths.data, [fileBase, '.mat']);

save(matPath, ...
    'SNR_dB', ...
    'numTrials', ...
    'timingMSE', ...
    'exactProbability', ...
    'cpSafeProbability', ...
    'detectionFailureRate', ...
    'selectedSNR', ...
    'timingErrorSamples', ...
    'Nfft', ...
    'Ncp', ...
    'D', ...
    'alpha', ...
    'elapsedTime');

fprintf('\nResults Saved\n');
fprintf('CSV: %s\n', csvPath);
fprintf('MAT: %s\n', matPath);