%% Experiment 02D
% BER vs Eb/N0 under Timing Offset
% Part 1: Simulation Setup and Data Generation
clear;
clc;
close all;

%% 1. Load Configuration
cfg = default_config();

Nfft = cfg.Nfft;
Ncp = cfg.Ncp;

M = cfg.M;
bitsPerSymbol = cfg.bitsPerSymbol;

rng(cfg.randomSeed);

%% 2. Simulation Parameters
EbN0dB = 0:2:12;

numDataSymbols = 1000;
numSymbols = numDataSymbols + 2;

symbolLength = Nfft + Ncp;

% Timing offset cases
tauCases = [0, -5, -5, 5];

% Whether ideal phase compensation is applied
useCompensation = [false, false, true, true];

caseNames = { ...
    'Perfect Timing', ...
    'Early Timing (-5)', ...
    'Early Timing (-5) + Compensation', ...
    'Late Timing (+5) + Compensation'};

numCases = length(tauCases);

%% 3. Display Configuration
fprintf('Experiment 02D \n');

fprintf('FFT Size: %d\n', Nfft);
fprintf('CP Length: %d\n', Ncp);
fprintf('Modulation Order: %d\n', M);

fprintf('Number of Data Symbols: %d\n', numDataSymbols);

fprintf('Eb/N0 Range: ');
fprintf('%d ', EbN0dB);
fprintf('\n');

fprintf('Number of Timing Cases: %d\n', numCases);

%% 4. Generate Random Bits
numBits = Nfft * bitsPerSymbol * numSymbols;

txBits = randi([0 1], numBits, 1);

%% 5. QPSK Modulation
bitGroups = reshape(txBits, bitsPerSymbol, []).';

symbolIndices = bi2de(bitGroups, 'left-msb');

modSymbols = pskmod(symbolIndices, M, pi/4);

%% 6. OFDM Modulation
X = reshape(modSymbols, Nfft, numSymbols);

x = ifft(X, Nfft, 1);

xCP = [x(end-Ncp+1:end, :); x];

txSignal = xCP(:);

%% 7. Verification
expectedSamples = numSymbols * symbolLength;

assert(length(txSignal) == expectedSamples, ...
    'OFDM signal length mismatch.');

assert(numCases == length(useCompensation), ...
    'Timing case configuration mismatch.');

fprintf('\nOFDM signal generation verified.\n');

fprintf('Total Generated Bits: %d\n', numBits);

fprintf('Total Signal Samples: %d\n', length(txSignal));

fprintf('\nTiming Cases:\n');

for i = 1:numCases

    fprintf('Case %d: %s\n', i, caseNames{i});

end

%% 8. AWGN Model Verification

fprintf('\nAWGN Model Verification\n');

% Use the same Eb/N0 convention as Experiment 01
% CP energy overhead is not included

testEbN0dB = 6;

% Convert Eb/N0 to sample SNR
testSNRdB = testEbN0dB + 10*log10(bitsPerSymbol);

fprintf('Test Eb/N0 = %.2f dB\n', testEbN0dB);
fprintf('Converted SNR = %.4f dB\n', testSNRdB);

%% 9. Add AWGN

rxSignal = add_awgn(txSignal, testSNRdB);

%% 10. Verify Noise Power

noise = rxSignal - txSignal;

signalPower = mean(abs(txSignal).^2);

measuredNoisePower = mean(abs(noise).^2);

measuredSNRdB = 10*log10( ...
    signalPower / measuredNoisePower);

fprintf('\nTarget SNR = %.4f dB\n', testSNRdB);
fprintf('Measured SNR = %.4f dB\n', measuredSNRdB);

SNRerror = abs(measuredSNRdB - testSNRdB);

fprintf('SNR Error = %.4f dB\n', SNRerror);

assert(SNRerror < 0.2, ...
    'AWGN SNR verification failed.');

fprintf('AWGN model verification passed.\n');

%% 11. Perfect Timing BER Verification
fprintf('\nerfect Timing BER Verification\n');

% Reference transmitted bits
bitsPerOFDM = Nfft * bitsPerSymbol;

txBitsReference = txBits( ...
    bitsPerOFDM+1 : end-bitsPerOFDM);

% Initialize BER arrays
BER_perfect = zeros(size(EbN0dB));
BER_theory = zeros(size(EbN0dB));

% Verify number of reference bits
assert(length(txBitsReference) == ...
    numDataSymbols*Nfft*bitsPerSymbol, ...
    'Reference bit length mismatch.');

for snrIdx = 1:length(EbN0dB)

    currentEbN0 = EbN0dB(snrIdx);

    % Convert Eb/N0 to sample SNR
    SNR_sample_dB = currentEbN0 ...
        + 10*log10(bitsPerSymbol);

    % Add AWGN
    rxSignal = add_awgn(txSignal, SNR_sample_dB);

    % Store received symbols
    Y_all = zeros(Nfft, numDataSymbols);

    %% Perfect timing FFT window extraction

    for symbolIdx = 2:numSymbols-1

        idealStart = ...
            (symbolIdx-1)*symbolLength + Ncp + 1;

        rxWindow = add_timing_offset( ...
            rxSignal, idealStart, 0, Nfft);

        Y_all(:, symbolIdx-1) = fft(rxWindow, Nfft);

    end

    %% QPSK demodulation

    rxSymbolIndices = pskdemod(Y_all(:), M, pi/4);

    rxBitGroups = de2bi( ...
        rxSymbolIndices, bitsPerSymbol, 'left-msb');

    rxBits = reshape(rxBitGroups.', [], 1);

    %% BER calculation

    numErrors = sum(rxBits ~= txBitsReference);

    BER_perfect(snrIdx) = ...
        numErrors / length(txBitsReference);

    %% Theoretical QPSK BER

    BER_theory(snrIdx) = ...
        0.5*erfc(sqrt(10^(currentEbN0/10)));

    fprintf('Eb/N0 = %2d dB | Errors = %5d | ', ...
        currentEbN0, numErrors);

    fprintf('Sim BER = %.6e | Theory BER = %.6e\n', ...
        BER_perfect(snrIdx), BER_theory(snrIdx));

end

fprintf('\nPerfect Timing BER simulation completed.\n');

%% 12. Four Timing Cases BER Comparison
fprintf('\nFour Timing Cases BER Comparison\n');

% Initialize BER matrix
% Rows: Timing cases
% Columns: Eb/N0 values

BER_cases = zeros(numCases, length(EbN0dB));

errorCount_cases = zeros(numCases, length(EbN0dB));

% Subcarrier indices
k = (0:Nfft-1).';

% Total number of evaluated bits
numBitsEvaluated = length(txBitsReference);

for snrIdx = 1:length(EbN0dB)

    currentEbN0 = EbN0dB(snrIdx);

    % Same Eb/N0 definition as Experiment 01
    SNR_sample_dB = currentEbN0 ...
        + 10*log10(bitsPerSymbol);

    % Generate one noisy signal for all four cases
    rxSignal = add_awgn(txSignal, SNR_sample_dB);

    for caseIdx = 1:numCases

        tau = tauCases(caseIdx);

        % Initialize received frequency-domain symbols
        Y_all = zeros(Nfft, numDataSymbols);

        %% FFT window extraction

        for symbolIdx = 2:numSymbols-1

            idealStart = ...
                (symbolIdx-1)*symbolLength + Ncp + 1;

            rxWindow = add_timing_offset( ...
                rxSignal, idealStart, tau, Nfft);

            Y_all(:, symbolIdx-1) = fft(rxWindow, Nfft);

        end

        %% Optional ideal phase compensation

        if useCompensation(caseIdx)

            phaseRotation = exp(1j*2*pi*k*tau/Nfft);

            Y_all = Y_all .* conj(phaseRotation);

        end

        %% QPSK demodulation

        rxSymbolIndices = pskdemod(Y_all(:), M, pi/4);

        rxBitGroups = de2bi( ...
            rxSymbolIndices, bitsPerSymbol, 'left-msb');

        rxBits = reshape(rxBitGroups.', [], 1);

        %% Calculate BER

        numErrors = sum(rxBits ~= txBitsReference);

        errorCount_cases(caseIdx, snrIdx) = numErrors;

        BER_cases(caseIdx, snrIdx) = ...
            numErrors / numBitsEvaluated;

    end

    %% Display results for current Eb/N0

    fprintf('\nEb/N0 = %2d dB\n', currentEbN0);

    for caseIdx = 1:numCases

        fprintf('  Case %d: BER = %.6e | Errors = %d\n', ...
            caseIdx, ...
            BER_cases(caseIdx, snrIdx), ...
            errorCount_cases(caseIdx, snrIdx));

    end

end

fprintf('\nFour timing cases simulation completed.\n');

%% 13. BER Comparison Verification
fprintf('\nBER Comparison Verification\n');

% Compare Case 1 with previous perfect timing results

fprintf('Perfect Timing BER comparison:\n');

for i = 1:length(EbN0dB)

    fprintf('Eb/N0 = %2d dB | Previous = %.6e | New = %.6e\n', ...
        EbN0dB(i), ...
        BER_perfect(i), ...
        BER_cases(1, i));

end

% Verify BER values are valid probabilities

assert(all(BER_cases(:) >= 0 & BER_cases(:) <= 1), ...
    'Invalid BER values detected.');

% Check approximate agreement between Case 1 and Case 3
% at lower Eb/N0 where error counts are sufficient

lowSNR = EbN0dB <= 6;

difference = abs( ...
    BER_cases(1, lowSNR) - BER_cases(3, lowSNR));

fprintf('\nMaximum BER difference between Case 1 and Case 3: %.6e\n', ...
    max(difference));

fprintf('BER comparison verification completed.\n');

%% 14. Plot BER vs Eb/N0
fprintf('\nPlotting BER Results\n');

figure('Color', 'w', 'Position', [100 100 900 600]);

hold on;

% BER cannot be displayed at zero on a logarithmic axis.
% Replace zero values only for visualization.
% Original BER data remain unchanged.

plotFloor = 1 / numBitsEvaluated;

BER_plot = BER_cases;
BER_plot(BER_plot == 0) = plotFloor;

% Case 1: Perfect Timing
semilogy(EbN0dB, BER_plot(1,:), 'bo-', ...
    'LineWidth', 1.8, 'MarkerSize', 7);

% Case 2: Early Timing without compensation
semilogy(EbN0dB, BER_plot(2,:), 'rs-', ...
    'LineWidth', 1.8, 'MarkerSize', 7);

% Case 3: Early Timing with compensation
semilogy(EbN0dB, BER_plot(3,:), 'g^-', ...
    'LineWidth', 1.8, 'MarkerSize', 7);

% Case 4: Late Timing with compensation
semilogy(EbN0dB, BER_plot(4,:), 'md-', ...
    'LineWidth', 1.8, 'MarkerSize', 7);

% Theoretical QPSK BER
semilogy(EbN0dB, BER_theory, 'k--', ...
    'LineWidth', 1.8);

grid on;
box on;

xlabel('E_b/N_0 (dB)', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);

title('BER Performance of QPSK-OFDM under Timing Offsets', ...
    'FontSize', 13);

legend( ...
    'Perfect Timing', ...
    'Early Timing (-5)', ...
    'Early Timing (-5) + Compensation', ...
    'Late Timing (+5) + Compensation', ...
    'Theoretical QPSK', ...
    'Location', 'southwest');

xticks(EbN0dB);

set(gca, 'YScale', 'log');
ylim([1e-6 1]);

set(gca, 'FontSize', 11);

fprintf('BER figure generated successfully.\n');

%% 15. Save BER Figure
fprintf('\nSaving Experiment 02D Results\n');

% Locate project root
experimentDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(experimentDir);

% Define output directories
figureDir = fullfile(projectRoot, 'results', 'figures');
dataDir = fullfile(projectRoot, 'results', 'data');
tableDir = fullfile(projectRoot, 'results', 'tables');

% Create directories if they do not exist
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
figurePath = fullfile( ...
    figureDir, 'exp02_ber_vs_ebn0.png');

exportgraphics(gcf, figurePath, 'Resolution', 300);

fprintf('Figure saved successfully.\n');
%% 16. Save MATLAB Data
dataPath = fullfile( ...
    dataDir, 'exp02_ber_vs_ebn0.mat');

save(dataPath, ...
    'EbN0dB', ...
    'BER_cases', ...
    'BER_theory', ...
    'BER_perfect', ...
    'errorCount_cases', ...
    'tauCases', ...
    'useCompensation', ...
    'caseNames', ...
    'numDataSymbols', ...
    'numBitsEvaluated', ...
    'cfg');

fprintf('MAT data saved successfully.\n');

%% 17. Save CSV Results
resultsTable = table( ...
    EbN0dB(:), ...
    BER_cases(1,:).', ...
    BER_cases(2,:).', ...
    BER_cases(3,:).', ...
    BER_cases(4,:).', ...
    BER_theory(:), ...
    errorCount_cases(1,:).', ...
    errorCount_cases(2,:).', ...
    errorCount_cases(3,:).', ...
    errorCount_cases(4,:).', ...
    'VariableNames', { ...
    'EbN0_dB', ...
    'BER_Perfect', ...
    'BER_Early_NoComp', ...
    'BER_Early_Comp', ...
    'BER_Late_Comp', ...
    'BER_Theory', ...
    'Errors_Perfect', ...
    'Errors_Early_NoComp', ...
    'Errors_Early_Comp', ...
    'Errors_Late_Comp'});

tablePath = fullfile( ...
    tableDir, 'exp02_ber_vs_ebn0.csv');

writetable(resultsTable, tablePath);

fprintf('CSV table saved successfully.\n');

%% 18. Verify Saved Files
assert(isfile(figurePath), ...
    'Figure file was not saved.');

assert(isfile(dataPath), ...
    'MAT file was not saved.');

assert(isfile(tablePath), ...
    'CSV file was not saved.');

fprintf('\nExperiment 02D Completed\n');

fprintf('Figure: %s\n', figurePath);
fprintf('Data:   %s\n', dataPath);
fprintf('Table:  %s\n', tablePath);

fprintf('\nAll Experiment 02D results saved successfully.\n');