%% Experiment 02B
% Timing Offset Modelling - Initial Verification
clear;
clc;
close all;

%% 1. Load configuration
cfg = default_config();

Nfft = cfg.Nfft;
Ncp = cfg.Ncp;

rng(cfg.randomSeed);

fprintf('FFT Size: %d\n', Nfft);
fprintf('CP Length: %d\n', Ncp);

%% 2. Generate three OFDM symbols
numSymbols = 3;

% Generate random QPSK symbols directly
dataSymbols = randi([0 3], Nfft, numSymbols);

X = pskmod(dataSymbols, 4, pi/4);

% IFFT
x = ifft(X, Nfft, 1);

% Add cyclic prefix
xCP = [x(end-Ncp+1:end, :); x];

% Convert to serial signal
txSignal = xCP(:);

%% 3. Define the ideal FFT window
symbolLength = Nfft + Ncp;

% Study the second OFDM symbol
targetSymbol = 2;

% MATLAB uses 1-based indexing
idealStart = (targetSymbol-1)*symbolLength + Ncp + 1;

fprintf('Ideal FFT Start: %d\n', idealStart);

%% 4. Extract FFT windows with different offsets
tauValues = [-5, 0, 5];

for i = 1:length(tauValues)

    tau = tauValues(i);

    rxWindow = add_timing_offset( ...
        txSignal, idealStart, tau, Nfft);

    fprintf('Offset = %+d, Window Length = %d\n', ...
        tau, length(rxWindow));
end

    %% 5. Frequency-domain analysis
    % Original frequency-domain symbols of Symbol 2
    X_original = X(:, targetSymbol);

    % Timing offsets to investigate
    tauValues = [-5, 0, 5];

    % Store FFT outputs
    Y_all = zeros(Nfft, length(tauValues));

    % Store errors
    magnitudeError = zeros(1, length(tauValues));
    frequencyError = zeros(1, length(tauValues));

    for i = 1:length(tauValues)

        tau = tauValues(i);

        % Extract FFT window
        rxWindow = add_timing_offset( ...
            txSignal, idealStart, tau, Nfft);

        % Perform FFT
        Y = fft(rxWindow, Nfft);

        % Store FFT result
        Y_all(:, i) = Y;

        % Calculate magnitude error
        magnitudeError(i) = ...
            mean(abs(abs(Y) - abs(X_original)));

        % Calculate frequency-domain error
        frequencyError(i) = ...
            mean(abs(Y - X_original).^2);

        fprintf('\nTiming Offset = %+d\n', tau);

        fprintf('Mean Magnitude Error = %.6e\n', ...
            magnitudeError(i));

        fprintf('Frequency-domain MSE = %.6e\n', ...
            frequencyError(i));

    end

    %% 6. Phase Rotation Verification
    fprintf('\nPhase Rotation Verification\n');

    % Select early timing offset
    tau = -5;

    % Extract FFT window
    rxWindow = add_timing_offset( ...
        txSignal, idealStart, tau, Nfft);

    % FFT
    Y = fft(rxWindow, Nfft);

    % Subcarrier indices
    k = (0:Nfft-1).';

    % Theoretical phase rotation
    phaseRotation = exp(1j * 2*pi*k*tau/Nfft);

    % Predicted frequency-domain signal
    Y_theory = X_original .* phaseRotation;

    % Compare actual FFT output with theoretical prediction
    theoryMSE = mean(abs(Y - Y_theory).^2);

    fprintf('Theory Verification MSE = %.6e\n', theoryMSE);

    %% 7. Ideal Phase Compensation
    % Remove the theoretical phase rotation
    Y_compensated = Y .* conj(phaseRotation);

    % Calculate MSE after compensation
    compensatedMSE = mean(abs(Y_compensated - X_original).^2);

    fprintf('MSE Before Compensation = %.6e\n', ...
        mean(abs(Y - X_original).^2));

    fprintf('MSE After Compensation  = %.6e\n', ...
        compensatedMSE);

    %% 8. Timing Offset Sensitivity Analysis
    fprintf('\nTiming Offset Sensitivity Analysis \n');

    % Timing offset range
    tauScan = -24:24;

    % Initialize result arrays
    magErrorScan = zeros(size(tauScan));
    freqMSEScan = zeros(size(tauScan));
    compensatedMSEScan = zeros(size(tauScan));

    % Subcarrier indices
    k = (0:Nfft-1).';

    for i = 1:length(tauScan)

        tau = tauScan(i);

        % Extract FFT window
        rxWindow = add_timing_offset( ...
            txSignal, idealStart, tau, Nfft);

        % FFT
        Y = fft(rxWindow, Nfft);

        % Magnitude error
        magErrorScan(i) = mean(abs(abs(Y) - abs(X_original)));

        % Frequency-domain MSE
        freqMSEScan(i) = mean(abs(Y - X_original).^2);

        % Ideal phase compensation
        phaseRotation = exp(1j * 2*pi*k*tau/Nfft);

        Y_compensated = Y .* conj(phaseRotation);

        % MSE after compensation
        compensatedMSEScan(i) = ...
            mean(abs(Y_compensated - X_original).^2);

    end

    fprintf('Timing offset scan completed.\n');
    fprintf('Number of offsets tested: %d\n', length(tauScan));

    %% 9. Plot Timing Offset Sensitivity
    figure('Color', 'w');

    plot(tauScan, freqMSEScan, ...
        'b-o', 'LineWidth', 1.5, 'MarkerSize', 3);

    hold on;

    plot(tauScan, compensatedMSEScan, ...
        'r-s', 'LineWidth', 1.5, 'MarkerSize', 3);

    % CP boundaries
    xline(-Ncp, '--k', 'CP Boundary', ...
        'LineWidth', 1.2);

    xline(0, '--g', 'Perfect Timing', ...
        'LineWidth', 1.2);

    xlabel('Timing Offset (samples)');
    ylabel('Frequency-domain MSE');

    title('OFDM Timing Offset Sensitivity');

    legend('Without Compensation', ...
        'With Ideal Phase Compensation', ...
        'Location', 'best');

    grid on;

    hold off;

    %% 10. Save Experiment Results
    % Get project root directory
    projectRoot = fileparts(fileparts(mfilename('fullpath')));

    % Define output directories
    figureDir = fullfile(projectRoot, 'results', 'figures');
    dataDir = fullfile(projectRoot, 'results', 'data');
    tableDir = fullfile(projectRoot, 'results', 'tables');

    % Create directories if necessary
    if ~exist(figureDir, 'dir')
        mkdir(figureDir);
    end

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    if ~exist(tableDir, 'dir')
        mkdir(tableDir);
    end

    % Save figure
    exportgraphics(gcf, ...
        fullfile(figureDir, 'exp02_timing_offset_sensitivity.png'), ...
        'Resolution', 300);

    % Save MATLAB data
    save(fullfile(dataDir, 'exp02_timing_offset_results.mat'), ...
        'tauScan', ...
        'magErrorScan', ...
        'freqMSEScan', ...
        'compensatedMSEScan', ...
        'Nfft', 'Ncp');

    % Create result table
    resultTable = table( ...
        tauScan(:), ...
        magErrorScan(:), ...
        freqMSEScan(:), ...
        compensatedMSEScan(:), ...
        'VariableNames', { ...
        'TimingOffset', ...
        'MagnitudeError', ...
        'FrequencyMSE', ...
        'CompensatedMSE'});

    % Save CSV
    writetable(resultTable, ...
        fullfile(tableDir, 'exp02_timing_offset_results.csv'));

    fprintf('\nExperiment 02 results saved successfully.\n');