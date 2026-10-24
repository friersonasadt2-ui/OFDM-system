%% Experiment 03H-4: Schmidl-Cox Multipath Monte Carlo
% Compare AWGN and fixed multipath channels
clear;
clc;
close all;

%% 1. Unified Experiment Configuration
cfg = default_config();

Nfft = cfg.Nfft;
Ncp = cfg.Ncp;
L = Nfft / 2;

D = cfg.arrivalDelay;
alpha = cfg.SC_threshold;

SNR_dB = cfg.SNR_dB;
numTrials = cfg.numTrials;

rng(cfg.randomSeed);

%% 2. Generate Schmidl-Cox Training Symbol
qpskData = randi([0 3], L, 1);
qpskSymbols = pskmod(qpskData, 4, pi/4);

X = zeros(Nfft, 1);
X(1:2:end) = qpskSymbols;

x = ifft(X, Nfft);
cp = x(end-Ncp+1:end);

trainingSymbol = [cp; x];

%% 3. Generate Data OFDM Symbol
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

%% 4. Construct Frame
txFrame = [trainingSymbol; dataOFDM];

txDelayed = [zeros(D,1); txFrame];

trueUsefulStart = D + Ncp + 1;

%% 5. Channel Configuration
channelNames = cfg.channelNames;
channelResponses = cfg.channelResponses;
numChannels = numel(channelNames);
assert(numChannels == numel(channelResponses), ...
    'Channel names and responses must have the same length.');

%% 6. Initialize Results
numSNR = length(SNR_dB);

timingMSE = NaN(numSNR, numChannels);

exactProbability = zeros(numSNR, numChannels);

cpSafeProbability = zeros(numSNR, numChannels);

noEstimateRate = zeros(numSNR, numChannels);

%% 7. Monte Carlo Simulation
simulationTimer = tic;

for channelIndex = 1:numChannels

    % Current channel impulse response
    h = channelResponses{channelIndex};

    % Maximum channel delay
    maxChannelDelay = length(h) - 1;

    % Apply channel
    rxClean = apply_multipath_channel(txDelayed, h);

    % Calculate received signal power over active frame
    signalPower = mean(abs(rxClean(D+1:D+numel(txFrame))).^2);
    assert(isfinite(signalPower) && signalPower > 0, ...
        'Invalid received signal power.');

    fprintf('\nChannel: %s\n', ...
        channelNames{channelIndex});

    for snrIndex = 1:numSNR

        currentSNR = SNR_dB(snrIndex);

        % Noise power for current SNR
        noisePower = signalPower / ...
            (10^(currentSNR/10));

        % Initialize trial results
        timingErrors = NaN(numTrials, 1);

        exactSuccess = false(numTrials, 1);

        cpSafeSuccess = false(numTrials, 1);

        detectionSuccess = false(numTrials, 1);

        %% Monte Carlo Trials
        for trial = 1:numTrials

            % Generate complex AWGN
            noise = sqrt(noisePower/2) * ...
                (randn(size(rxClean)) + ...
                 1j*randn(size(rxClean)));

            % Received noisy signal
            rxNoisy = rxClean + noise;

            % Schmidl-Cox timing metric
            [~, ~, M] = ...
                schmidl_cox_metric(rxNoisy, L);

            % Timing estimation
            [estimatedStart, info] = ...
                schmidl_cox_estimator(M, alpha);

            % Record timing result
            if info.detected && isfinite(estimatedStart)

                detectionSuccess(trial) = true;

                timingErrors(trial) = ...
                    estimatedStart - trueUsefulStart;

                exactSuccess(trial) = ...
                    (timingErrors(trial) == 0);

                % Channel-dependent CP-safe region
                cpSafeSuccess(trial) = ...
                    (timingErrors(trial) >= ...
                    -(Ncp-maxChannelDelay)) && ...
                    (timingErrors(trial) <= 0);

            end

        end  % End trials

        %% Calculate performance metrics
        validTrials = detectionSuccess;

        if any(validTrials)

            timingMSE(snrIndex, channelIndex) = ...
                mean(timingErrors(validTrials).^2);

        end

        exactProbability(snrIndex, channelIndex) = ...
            mean(exactSuccess);

        cpSafeProbability(snrIndex, channelIndex) = ...
            mean(cpSafeSuccess);

        noEstimateRate(snrIndex, channelIndex) = ...
            mean(~detectionSuccess);

        %% Display current results
        fprintf(['SNR = %2d dB | MSE = %8.3f | ', ...
                 'Exact = %.3f | CP-safe = %.3f | ', ...
                 'No-estimate = %.3f\n'], ...
                 currentSNR, ...
                 timingMSE(snrIndex, channelIndex), ...
                 exactProbability(snrIndex, channelIndex), ...
                 cpSafeProbability(snrIndex, channelIndex), ...
                 noEstimateRate(snrIndex, channelIndex));

    end  % End SNR sweep

end  % End channel loop

%% 8. Simulation Runtime
elapsedTime = toc(simulationTimer);

totalTrials = numTrials * numSNR * numChannels;

fprintf('\nSimulation Runtime\n');

fprintf('Total trials: %d\n', totalTrials);

fprintf('Elapsed time: %.2f seconds\n', elapsedTime);

fprintf('Average time per trial: %.6f seconds\n', ...
    elapsedTime / totalTrials);

disp('Multipath Monte Carlo simulation completed.');

%% 9. Save Monte Carlo Results
% Get centralized project paths
paths = project_paths();

% Create results table
resultsTable = table( ...
    SNR_dB(:), ...
    timingMSE(:,1), ...
    timingMSE(:,2), ...
    exactProbability(:,1), ...
    exactProbability(:,2), ...
    cpSafeProbability(:,1), ...
    cpSafeProbability(:,2), ...
    noEstimateRate(:,1), ...
    noEstimateRate(:,2), ...
    'VariableNames', { ...
    'SNR_dB', ...
    'AWGN_MSE', ...
    'Multipath_MSE', ...
    'AWGN_Exact', ...
    'Multipath_Exact', ...
    'AWGN_CP_Safe', ...
    'Multipath_CP_Safe', ...
    'AWGN_No_Estimate', ...
    'Multipath_No_Estimate'});

% Define output filenames
fileBase = sprintf('sc_awgn_vs_multipath_%d', numTrials);

% Save CSV
csvPath = fullfile(paths.tables, [fileBase, '.csv']);

writetable(resultsTable, csvPath);

% Save MAT
matPath = fullfile(paths.data, [fileBase, '.mat']);

save(matPath, ...
    'SNR_dB', ...
    'numTrials', ...
    'channelNames', ...
    'channelResponses', ...
    'timingMSE', ...
    'exactProbability', ...
    'cpSafeProbability', ...
    'noEstimateRate', ...
    'Nfft', ...
    'Ncp', ...
    'D', ...
    'alpha', ...
    'elapsedTime');

fprintf('\nResults Saved\n');

fprintf('CSV: %s\n', csvPath);

fprintf('MAT: %s\n', matPath);