%% Experiment 03F: Schmidl-Cox Performance Visualization
% Plot Monte Carlo results from saved CSV
clear;
clc;
close all;

%% 1. Initialize project paths
paths = project_paths();

%% 2. Load Monte Carlo results
csvPath = fullfile(paths.tables, ...
    'sc_awgn_monte_carlo_10000.csv');

assert(isfile(csvPath), ...
    'Monte Carlo CSV file not found.');

results = readtable(csvPath);

%% 3. Extract performance metrics
SNR_dB = results.SNR_dB;

timingMSE = results.Timing_MSE;

exactProbability = results.Exact_Probability;

cpSafeProbability = results.CP_Safe_Probability;

%% 4. Plot Timing MSE vs SNR
figure;

plot(SNR_dB, timingMSE, '-o', ...
    'LineWidth', 1.8, ...
    'MarkerSize', 6);

grid on;

xlabel('SNR (dB)');

ylabel('Timing MSE (samples^2)');

title('Schmidl-Cox Timing MSE vs SNR');

xticks(SNR_dB);

%% 5. Save Figure
figPath = fullfile(paths.figures, ...
    'sc_awgn_timing_mse.png');

exportgraphics(gcf, figPath, ...
    'Resolution', 300);

fprintf('Timing MSE figure saved:\n%s\n', figPath);

%% 6. Plot Exact Timing Probability vs SNR
figure;

plot(SNR_dB, exactProbability, '-o', ...
    'LineWidth', 1.8, ...
    'MarkerSize', 6);

grid on;

xlabel('SNR (dB)');

ylabel('Exact Timing Probability');

title('Schmidl-Cox Exact Timing Probability vs SNR');

xticks(SNR_dB);

ylim([0 0.12]);
yticks(0:0.02:0.12);

%% Save Figure
figPath = fullfile(paths.figures, ...
    'sc_awgn_exact_probability.png');

exportgraphics(gcf, figPath, ...
    'Resolution', 300);

fprintf('Exact Timing Probability figure saved:\n%s\n', figPath);

%% 7. Plot CP-safe Timing Probability vs SNR
figure;

plot(SNR_dB, cpSafeProbability, '-o', ...
    'LineWidth', 1.8, ...
    'MarkerSize', 6);

grid on;

xlabel('SNR (dB)');

ylabel('CP-safe Timing Probability');

title('Schmidl-Cox CP-safe Timing Probability vs SNR');

xticks(SNR_dB);

ylim([0 1]);

%% Save Figure
figPath = fullfile(paths.figures, ...
    'sc_awgn_cp_safe_probability.png');

exportgraphics(gcf, figPath, ...
    'Resolution', 300);

fprintf('CP-safe Timing Probability figure saved:\n%s\n', figPath);