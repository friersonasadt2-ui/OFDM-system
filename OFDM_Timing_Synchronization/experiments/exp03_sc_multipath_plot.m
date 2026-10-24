%% Experiment 03H-5: AWGN vs Multipath Performance Plots
clear;
clc;
close all;

%% 1. Get Project Paths
paths = project_paths();

%% 2. Load Saved Monte Carlo Results
dataPath = fullfile( ...
    paths.data, ...
    'sc_awgn_vs_multipath_10000.mat');

load(dataPath, ...
    'SNR_dB', ...
    'timingMSE', ...
    'exactProbability', ...
    'cpSafeProbability');

fprintf('Monte Carlo data loaded successfully.\n');

%% 3. Plot Timing MSE
figure('Color', 'w');

plot(SNR_dB, timingMSE(:,1), '-o', ...
    'LineWidth', 1.8, 'MarkerSize', 6);

hold on;

plot(SNR_dB, timingMSE(:,2), '--s', ...
    'LineWidth', 1.8, 'MarkerSize', 6);

grid on;

xlabel('SNR (dB)');
ylabel('Timing MSE (samples^2)');
title('Schmidl-Cox Timing MSE: AWGN vs Multipath');

legend('AWGN', 'Fixed Multipath', ...
    'Location', 'northeast');

xticks(SNR_dB);

exportgraphics(gcf, ...
    fullfile(paths.figures, 'sc_awgn_vs_multipath_mse.png'), ...
    'Resolution', 300);

%% 4. Plot Exact Timing Probability
figure('Color', 'w');

plot(SNR_dB, exactProbability(:,1), '-o', ...
    'LineWidth', 1.8, 'MarkerSize', 6);

hold on;

plot(SNR_dB, exactProbability(:,2), '--s', ...
    'LineWidth', 1.8, 'MarkerSize', 6);

grid on;

xlabel('SNR (dB)');
ylabel('Exact Timing Probability');
title('Schmidl-Cox Exact Timing Probability');

legend('AWGN', 'Fixed Multipath', ...
    'Location', 'northwest');

xticks(SNR_dB);
ylim([0 0.2]);

exportgraphics(gcf, ...
    fullfile(paths.figures, 'sc_awgn_vs_multipath_exact.png'), ...
    'Resolution', 300);

%% 5. Plot CP-safe Probability
figure('Color', 'w');

plot(SNR_dB, cpSafeProbability(:,1), '-o', ...
    'LineWidth', 1.8, 'MarkerSize', 6);

hold on;

plot(SNR_dB, cpSafeProbability(:,2), '--s', ...
    'LineWidth', 1.8, 'MarkerSize', 6);

grid on;

xlabel('SNR (dB)');
ylabel('CP-safe Timing Probability');
title('Schmidl-Cox CP-safe Probability');

legend('AWGN', 'Fixed Multipath', ...
    'Location', 'southeast');

xticks(SNR_dB);
ylim([0 1.05]);

exportgraphics(gcf, ...
    fullfile(paths.figures, 'sc_awgn_vs_multipath_cp_safe.png'), ...
    'Resolution', 300);

%% 6. Completion Message
fprintf('\nThree figures saved successfully.\n');
fprintf('Directory: %s\n', paths.figures);