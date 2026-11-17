%% Experiment 03G: Schmidl-Cox Timing Error Distribution
clear;
clc;
close all;

%% 1. Initialize Project Paths
paths = project_paths();

%% 2. Load Monte Carlo Data
matPath = fullfile(paths.data, ...
    'sc_awgn_monte_carlo_10000.mat');

assert(isfile(matPath), ...
    'Monte Carlo MAT file not found.');

data = load(matPath);

%% 3. Extract Timing Error Samples
selectedSNR = data.selectedSNR;

timingErrorSamples = data.timingErrorSamples;

%% 4. Plot Timing Error Distributions
% Create a larger figure
fig = figure('Color', 'w', ...
    'Position', [100, 100, 1100, 1000]);

% Create tiled layout with better spacing
t = tiledlayout(3, 1, ...
    'TileSpacing', 'loose', ...
    'Padding', 'loose');

for k = 1:length(selectedSNR)

    % Create subplot
    ax = nexttile;

    % Extract valid timing errors
    errors = timingErrorSamples(:, k);
    errors = errors(isfinite(errors));

    % Plot histogram
    histogram(ax, errors, ...
        'BinMethod', 'integers', ...
        'Normalization', 'probability', ...
        'FaceColor', [0.2 0.5 0.8]);

    hold(ax, 'on');

    referenceError = -data.Ncp / 2;

    xline(ax, referenceError, '--r', ...
        'LineWidth', 1.5);
    
    % Grid and labels
    grid(ax, 'on');

    xlabel(ax, 'Timing Error (samples)', ...
        'FontSize', 11);

    ylabel(ax, 'Probability', ...
        'FontSize', 11);

    title(ax, sprintf('SNR = %d dB', selectedSNR(k)), ...
        'FontSize', 13);

    % Set font size
    ax.FontSize = 11;

    hold(ax, 'off');

end

% Overall title
title(t, 'Schmidl-Cox Timing Error Distribution', ...
    'FontSize', 16, ...
    'FontWeight', 'bold');

%% 5. Save Figure
figPath = fullfile(paths.figures, ...
    'sc_awgn_timing_error_distribution.png');

exportgraphics(fig, figPath, ...
    'Resolution', 300);

fprintf('\nTiming error distribution saved:\n%s\n', figPath);