%% Experiment 03H-1: Schmidl-Cox Multipath Basic Test
% Compare timing metrics under AWGN and fixed multipath with AWGN
clear;
clc;
close all;

%% 1. Unified Experiment Configuration
cfg = default_config();
Nfft = cfg.Nfft;
Ncp = cfg.Ncp;
L = Nfft / 2;
D = cfg.arrivalDelay;
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

%% Normalize Both OFDM Symbols to the Same Energy
trainingSymbol = normalize_training_energy(trainingSymbol, cfg.trainingEnergy);
dataOFDM = normalize_training_energy(dataOFDM, cfg.trainingEnergy);

%% 4. Construct Transmitted Frame
txFrame = [trainingSymbol; dataOFDM];

txDelayed = [zeros(D,1); txFrame];

trueUsefulStart = D + Ncp + 1;

%% 5. Define Multipath Channel
h = cfg.hMultipath;

maxChannelDelay = length(h) - 1;

%% 6. Apply Channels
% Ideal channel
rxIdeal = txDelayed;

% Multipath channel
rxMultipath = apply_multipath_channel(txDelayed, h);

%% 6A. Add AWGN to Both Channels

SNR_dB = cfg.exampleSNR_dB;

% Compute signal power over the active received frame
activeIdx = D+1 : D+numel(txFrame);
powerIdeal = mean(abs(rxIdeal(activeIdx)).^2);
powerMultipath = mean(abs(rxMultipath(activeIdx)).^2);

% Noise power for the target SNR
noisePowerIdeal = powerIdeal / (10^(SNR_dB/10));

noisePowerMultipath = powerMultipath / (10^(SNR_dB/10));

% Generate independent complex AWGN
noiseIdeal = sqrt(noisePowerIdeal/2) * ...
    (randn(size(rxIdeal)) + 1j*randn(size(rxIdeal)));

noiseMultipath = sqrt(noisePowerMultipath/2) * ...
    (randn(size(rxMultipath)) + 1j*randn(size(rxMultipath)));

% Noisy received signals
rxIdeal = rxIdeal + noiseIdeal;

rxMultipath = rxMultipath + noiseMultipath;

fprintf('\nAWGN added at target SNR = %.1f dB\n', SNR_dB);
%% 7. Compute Schmidl-Cox Timing Metrics
[~, ~, M_ideal] = schmidl_cox_metric(rxIdeal, L);

[~, ~, M_multipath] = ...
    schmidl_cox_metric(rxMultipath, L);

%% 7A. Timing Estimation

alpha = cfg.SC_threshold;

% Ideal channel timing estimation
[estimatedIdeal, infoIdeal] = ...
    schmidl_cox_estimator(M_ideal, alpha);

% Multipath channel timing estimation
[estimatedMultipath, infoMultipath] = ...
    schmidl_cox_estimator(M_multipath, alpha);

% Timing errors
errorIdeal = estimatedIdeal - trueUsefulStart;

errorMultipath = estimatedMultipath - trueUsefulStart;

% CP-safe regions
% Ideal channel: [-Ncp, 0]
% Multipath channel: [-(Ncp-maxChannelDelay), 0]

cpSafeIdeal = ...
    (errorIdeal >= -Ncp) && (errorIdeal <= 0);

cpSafeMultipath = ...
    (errorMultipath >= -(Ncp-maxChannelDelay)) && ...
    (errorMultipath <= 0);

%% Display Timing Estimation Results

fprintf('\nTiming Estimation Results\n');

fprintf('AWGN estimated start: %d\n', estimatedIdeal);
fprintf('AWGN timing error: %d samples\n', errorIdeal);
fprintf('AWGN CP-safe: %d\n', cpSafeIdeal);

fprintf('Multipath estimated start: %d\n', estimatedMultipath);
fprintf('Multipath timing error: %d samples\n', errorMultipath);
fprintf('Multipath CP-safe: %d\n', cpSafeMultipath);

%% 8. Display Results
fprintf('\nExperiment 03H-1: Multipath Basic Test\n');

fprintf('FFT size: %d\n', Nfft);

fprintf('CP length: %d\n', Ncp);

fprintf('True useful start: %d\n', trueUsefulStart);

fprintf('Channel taps:\n');

disp(h.');

fprintf('Maximum channel delay: %d samples\n', ...
    maxChannelDelay);

fprintf('AWGN metric maximum: %.6f\n', max(M_ideal));

fprintf('Multipath metric maximum: %.6f\n', ...
    max(M_multipath));

%% 9. Plot Timing Metrics
figure('Color', 'w', ...
    'Position', [100, 100, 1000, 500]);

plot(1:length(M_ideal), M_ideal, ...
    'LineWidth', 1.8);

hold on;

plot(1:length(M_multipath), M_multipath, ...
    '--', 'LineWidth', 1.8);

xline(trueUsefulStart, ':k', ...
    'LineWidth', 1.5);

grid on;

xlabel('Candidate Timing Index');

ylabel('Schmidl-Cox Timing Metric');

title(sprintf('Schmidl-Cox Timing Metric: AWGN vs Multipath (SNR = %g dB)', SNR_dB));

legend('AWGN Channel', ...
       'Multipath + AWGN Channel', ...
       'True FFT Start', ...
       'Location', 'best');

xlim([D-10, D+Ncp+30]);

%% 10. Save Figure
paths = project_paths();

figPath = fullfile(paths.figures, ...
    'sc_awgn_vs_multipath_metric.png');

exportgraphics(gcf, figPath, ...
    'Resolution', 300);

fprintf('\nFigure saved:\n%s\n', figPath);