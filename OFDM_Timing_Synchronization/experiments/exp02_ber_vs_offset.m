%% Experiment 02C
% BER vs Timing Offset
% Part 1: OFDM Data Generation

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
numDataSymbols = 1000;

% Two additional symbols for boundary protection
numSymbols = numDataSymbols + 2;

symbolLength = Nfft + Ncp;

fprintf('Experiment 02C\n');

fprintf('FFT Size: %d\n', Nfft);
fprintf('CP Length: %d\n', Ncp);
fprintf('Number of Data Symbols: %d\n', numDataSymbols);

%% 3. Generate Random Bits
numBits = Nfft * bitsPerSymbol * numSymbols;

txBits = randi([0 1], numBits, 1);

fprintf('Total Generated Bits: %d\n', numBits);

%% 4. QPSK Modulation
% Convert bits into groups
bitGroups = reshape(txBits, bitsPerSymbol, []).';

% Convert binary groups to decimal symbols
symbolIndices = bi2de(bitGroups, 'left-msb');

% QPSK modulation
modSymbols = pskmod(symbolIndices, M, pi/4);

%% 5. OFDM Modulation
% Arrange QPSK symbols into OFDM blocks
X = reshape(modSymbols, Nfft, numSymbols);

% IFFT
x = ifft(X, Nfft, 1);

% Add Cyclic Prefix
xCP = [x(end-Ncp+1:end, :); x];

% Parallel to Serial
txSignal = xCP(:);

fprintf('Generated OFDM Symbols: %d\n', numSymbols);
fprintf('Total Signal Samples: %d\n', length(txSignal));

%% 6. Basic Verification
expectedSamples = numSymbols * symbolLength;

assert(length(txSignal) == expectedSamples, ...
    'OFDM signal length mismatch.');

fprintf('OFDM signal generation verified successfully.\n');

%% 7. BER Verification under Timing Offset
fprintf('\nBER Verification\n');

% Timing offset sensitivity range
tauValues = -24:24;

% Subcarrier indices
k = (0:Nfft-1).';

% Initialize BER arrays
BER_noComp = zeros(size(tauValues));
BER_comp = zeros(size(tauValues));

% Reference transmitted symbols
% Exclude the first and last OFDM symbols
X_reference = X(:, 2:end-1);

% Convert transmitted symbols back to bits
txSymbolIndices = pskdemod(X_reference(:), M, pi/4);

txBitGroups = de2bi(txSymbolIndices, bitsPerSymbol, 'left-msb');

txBitsReference = reshape(txBitGroups.', [], 1);

for i = 1:length(tauValues)

    tau = tauValues(i);

    % Initialize received frequency-domain symbols
    Y_all = zeros(Nfft, numDataSymbols);

    %% Extract FFT windows

    for symbolIdx = 2:numSymbols-1

        % Ideal FFT start position
        idealStart = ...
            (symbolIdx-1)*symbolLength + Ncp + 1;

        % Apply timing offset
        rxWindow = add_timing_offset( ...
            txSignal, idealStart, tau, Nfft);

        % FFT
        Y = fft(rxWindow, Nfft);

        % Store frequency-domain symbols
        Y_all(:, symbolIdx-1) = Y;

    end

    %% BER without compensation

    rxSymbolIndices = pskdemod(Y_all(:), M, pi/4);

    rxBitGroups = de2bi(rxSymbolIndices, ...
        bitsPerSymbol, 'left-msb');

    rxBits = reshape(rxBitGroups.', [], 1);

    BER_noComp(i) = ...
        sum(rxBits ~= txBitsReference) / length(txBitsReference);

    %% Ideal phase compensation

    phaseRotation = exp(1j * 2*pi*k*tau/Nfft);

    Y_comp = Y_all .* conj(phaseRotation);

    rxSymbolIndicesComp = pskdemod(Y_comp(:), M, pi/4);

    rxBitGroupsComp = de2bi(rxSymbolIndicesComp, ...
        bitsPerSymbol, 'left-msb');

    rxBitsComp = reshape(rxBitGroupsComp.', [], 1);

    BER_comp(i) = ...
        sum(rxBitsComp ~= txBitsReference) / length(txBitsReference);

    %% Display results

    fprintf('\nTiming Offset = %+d\n', tau);

    fprintf('BER without compensation = %.6f\n', ...
        BER_noComp(i));

    fprintf('BER with compensation    = %.6f\n', ...
        BER_comp(i));

end

%% 8. BER Results Summary
fprintf('\nBER vs Timing Offset Summary\n');

BER_results = table( ...
    tauValues(:), ...
    BER_noComp(:), ...
    BER_comp(:), ...
    'VariableNames', { ...
    'TimingOffset', ...
    'BER_WithoutCompensation', ...
    'BER_WithCompensation'});

disp(BER_results);

% Verify ideal compensation within CP
cpRegion = (tauValues >= -Ncp) & (tauValues <= 0);

assert(all(BER_comp(cpRegion) == 0), ...
    'Unexpected bit errors within CP protection region.');

fprintf('CP protection region BER verification passed.\n');

%% 9. Plot BER vs Timing Offset
figure('Color', 'w');

plot(tauValues, BER_noComp, ...
    'b-o', 'LineWidth', 1.5, 'MarkerSize', 3);

hold on;

plot(tauValues, BER_comp, ...
    'r-s', 'LineWidth', 1.5, 'MarkerSize', 3);

% CP boundary
xline(-Ncp, '--k', 'CP Boundary', ...
    'LineWidth', 1.2);

% Perfect timing
xline(0, '--g', 'Perfect Timing', ...
    'LineWidth', 1.2);

xlabel('Timing Offset (samples)');
ylabel('Bit Error Rate (BER)');

title('BER vs Timing Offset in QPSK-OFDM');

legend('Without Compensation', ...
    'With Ideal Phase Compensation', ...
    'Location', 'best');

grid on;

ylim([0 0.6]);

hold off;

%% 10. Save BER Experiment Results
% Get project root
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% Output directories
figureDir = fullfile(projectRoot, 'results', 'figures');
dataDir = fullfile(projectRoot, 'results', 'data');
tableDir = fullfile(projectRoot, 'results', 'tables');

% Create directories
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
    fullfile(figureDir, 'exp02_ber_vs_timing_offset.png'), ...
    'Resolution', 300);

% Save MATLAB data
save(fullfile(dataDir, 'exp02_ber_vs_offset.mat'), ...
    'tauValues', ...
    'BER_noComp', ...
    'BER_comp', ...
    'Nfft', ...
    'Ncp', ...
    'numDataSymbols');

% Save CSV
writetable(BER_results, ...
    fullfile(tableDir, 'exp02_ber_vs_offset.csv'));

fprintf('\nBER experiment results saved successfully.\n');