%% Experiment 01: OFDM Baseline - BER vs Eb/N0
% Objectives:
% 1. Simulate a QPSK-OFDM system over an AWGN channel.
% 2. Calculate BER under different Eb/N0 values.
% 3. Compare simulated BER with theoretical QPSK BER.
% 4. Save experimental results and figures.
clear;
clc;
close all;

%% 1. Initialize project
% Locate project root
experimentDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(experimentDir);

% Add source code folders
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(fullfile(projectRoot, 'config'));

% Load configuration
cfg = default_config();

% Set random seed
rng(cfg.randomSeed);

fprintf('Experiment 01: OFDM Baseline\n');

%% 2. Simulation parameters

Nfft = cfg.Nfft;
Ncp = cfg.Ncp;

M = cfg.M;
bitsPerSymbol = cfg.bitsPerSymbol;

% Interpret the configured sweep as Eb/N0 for this experiment
EbN0_dB = cfg.SNR_dB;

numOFDMSymbols = cfg.numOFDMSymbols;

% Number of transmitted bits
numBits = Nfft * bitsPerSymbol * numOFDMSymbols;

fprintf('FFT size: %d\n', Nfft);
fprintf('CP length: %d\n', Ncp);
fprintf('Modulation order: %d\n', M);
fprintf('Number of OFDM symbols: %d\n', numOFDMSymbols);
fprintf('Number of bits: %d\n\n', numBits);

%% 3. Generate random bits

bits = randi([0 1], numBits, 1);

%% 4. QPSK modulation

txSymbols = qpsk_mod(bits);

%% 5. OFDM modulation

txSignal = ofdm_modulate(txSymbols, cfg);

%% 6. Initialize BER results

BER_sim = zeros(size(EbN0_dB));
BER_theory = zeros(size(EbN0_dB));

numErrors = zeros(size(EbN0_dB));

%% 7. Monte Carlo simulation over Eb/N0

fprintf('Starting BER simulation...\n\n');

for i = 1:length(EbN0_dB)

    currentEbN0 = EbN0_dB(i);

    % Convert Eb/N0 to time-domain sample SNR
    % All Nfft subcarriers carry QPSK data
    % CP consumes energy but does not carry additional information
    % SNR_sample = Eb/N0 * bitsPerSymbol * Nfft/(Nfft+Ncp)

    SNR_sample_dB = currentEbN0 ...
    + 10*log10(bitsPerSymbol);

    % Add AWGN
    rxSignal = add_awgn(txSignal, SNR_sample_dB);

    % OFDM demodulation
    rxSymbols = ofdm_demodulate(rxSignal, cfg);

    % QPSK demodulation
    bits_hat = qpsk_demod(rxSymbols);

    % Count bit errors
    numErrors(i) = sum(bits ~= bits_hat);

    % Calculate simulated BER
    BER_sim(i) = numErrors(i) / numBits;

    % Theoretical BER of coherent Gray-coded QPSK
    BER_theory(i) = 0.5 * erfc(sqrt(10^(currentEbN0/10)));

    fprintf('Eb/N0 = %2d dB | Errors = %6d | BER = %.6e\n', ...
        currentEbN0, numErrors(i), BER_sim(i));

end

fprintf('\nSimulation completed.\n');

%% 8. Plot BER vs Eb/N0

figure('Color', 'w');

semilogy(EbN0_dB, BER_sim, 'bo-', ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7);

hold on;

semilogy(EbN0_dB, BER_theory, 'r--', ...
    'LineWidth', 1.8);

grid on;

xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);

title('BER Performance of QPSK-OFDM over AWGN', ...
    'FontSize', 13);

legend('OFDM Simulation', ...
       'Theoretical QPSK', ...
       'Location', 'southwest');

ylim([1e-6 1]);

set(gca, 'FontSize', 11);

%% 9. Save experimental results

% Create output directories if needed
figureDir = fullfile(projectRoot, 'results', 'figures');
dataDir = fullfile(projectRoot, 'results', 'data');
tableDir = fullfile(projectRoot, 'results', 'tables');

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
    fullfile(figureDir, 'exp01_ber_vs_ebn0.png'), ...
    'Resolution', 300);

% Save MATLAB data
save(fullfile(dataDir, 'exp01_results.mat'), ...
    'EbN0_dB', ...
    'BER_sim', ...
    'BER_theory', ...
    'numErrors', ...
    'numBits', ...
    'cfg');

% Save CSV table
resultsTable = table( ...
    EbN0_dB(:), ...
    BER_sim(:), ...
    BER_theory(:), ...
    numErrors(:), ...
    'VariableNames', ...
    {'EbN0_dB', 'BER_Simulation', ...
     'BER_Theory', 'BitErrors'});

writetable(resultsTable, ...
    fullfile(tableDir, 'exp01_ber_results.csv'));

fprintf('\nResults saved successfully.\n');
fprintf('Figure: results/figures/exp01_ber_vs_ebn0.png\n');
fprintf('Data:   results/data/exp01_results.mat\n');
fprintf('Table:  results/tables/exp01_ber_results.csv\n');