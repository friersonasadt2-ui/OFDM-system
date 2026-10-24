%% Experiment 03: Schmidl-Cox Noiseless Timing Metric
% Objective:
% Demonstrate the plateau effect of the Schmidl-Cox
% timing metric under ideal noiseless conditions.
clear;
clc;
close all;

%% 1. Initialize Parameters
cfg = default_config();

Nfft = cfg.Nfft;
Ncp = cfg.Ncp;

L = Nfft / 2;
D = cfg.arrivalDelay;

rng(cfg.randomSeed);

fprintf('S&C Noiseless Timing Metric\n');

fprintf('FFT Size: %d\n', Nfft);
fprintf('CP Length: %d\n', Ncp);
fprintf('Repeated Half Length: %d\n', L);
fprintf('Arrival Delay: %d\n', D);

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

%% 4. Normalize Symbol Energy
trainingSymbol = normalize_training_energy( ...
    trainingSymbol, cfg.trainingEnergy);

dataOFDM = normalize_training_energy( ...
    dataOFDM, cfg.trainingEnergy);

%% 5. Construct Noiseless Received Signal
txFrame = [trainingSymbol; dataOFDM];

rxNoiseless = [zeros(D,1); txFrame];

trueUsefulStart = D + Ncp + 1;

fprintf('True Useful Symbol Start: %d\n', ...
    trueUsefulStart);

fprintf('No AWGN added.\n');
fprintf('No multipath channel applied.\n');

%% 6. Verify Repeated-Half Structure
usefulTraining = trainingSymbol(Ncp+1:end);

firstHalf = usefulTraining(1:L);
secondHalf = usefulTraining(L+1:end);

repetitionError = max(abs(firstHalf-secondHalf));

fprintf('Repeated-Half Error: %.6e\n', ...
    repetitionError);

assert(repetitionError < 1e-10, ...
    'S&C repeated-half structure verification failed.');

fprintf('Training symbol verification passed.\n');

%% 7. Calculate Noiseless Schmidl-Cox Timing Metric
[P, R, M] = schmidl_cox_metric(rxNoiseless, L);

candidateIndex = (1:numel(M)).';

fprintf('\nNoiseless Timing Metric Verification\n');

fprintf('Number of candidate timing positions: %d\n', ...
    numel(M));

[maxMetric, maxIndex] = max(M);

fprintf('Maximum Timing Metric: %.8f\n', maxMetric);
fprintf('Maximum Metric Index: %d\n', maxIndex);

%% 8. Check the CP Plateau
plateauStart = D + 1;
plateauEnd = trueUsefulStart;

plateauIndices = (plateauStart:plateauEnd).';

plateauValues = M(plateauIndices);

fprintf('\nExpected CP plateau range: [%d, %d]\n', ...
    plateauStart, plateauEnd);

fprintf('Minimum metric in plateau: %.8f\n', ...
    min(plateauValues));

fprintf('Maximum metric in plateau: %.8f\n', ...
    max(plateauValues));

fprintf('Mean metric in plateau: %.8f\n', ...
    mean(plateauValues));

fprintf('Plateau variation: %.8e\n', ...
    max(plateauValues)-min(plateauValues));

%% 9. Display Metric Values Around the Plateau
fprintf('\nTiming Metric Around CP Plateau:\n');

checkIndices = (D-5):(trueUsefulStart+5);

for d = checkIndices

    if d >= 1 && d <= numel(M)

        fprintf('d = %3d, M(d) = %.8f\n', ...
            d, M(d));

    end

end

%% 10. Plot Noiseless Schmidl-Cox Timing Metric
figure('Color', 'w', 'Position', [100 100 1000 550]);

plot(candidateIndex, M, ...
    'b-', 'LineWidth', 1.8);

hold on;

% Mark the true useful symbol start
xline(trueUsefulStart, 'r--', ...
    'LineWidth', 1.5, ...
    'Label', 'True Useful Start');

% Mark the CP plateau boundaries
xline(plateauStart, 'k:', ...
    'LineWidth', 1.2);

xline(plateauEnd, 'k:', ...
    'LineWidth', 1.2);

xlabel('Candidate Timing Index d', ...
    'FontSize', 12);

ylabel('Schmidl-Cox Timing Metric M(d)', ...
    'FontSize', 12);

title('Schmidl-Cox Timing Metric under Noiseless Conditions', ...
    'FontSize', 14);

grid on;
box on;

ylim([0 1.1]);

legend('Timing Metric', ...
    'True Useful Symbol Start', ...
    'CP Plateau Boundary', ...
    'Location', 'best');

hold off;

%% 11. Plot Zoomed CP Plateau
figure('Color', 'w', 'Position', [150 150 1000 550]);

plot(candidateIndex, M, ...
    'b-', 'LineWidth', 2);

hold on;

% Highlight CP plateau region
xline(plateauStart, 'k--', ...
    'LineWidth', 1.3);

xline(trueUsefulStart, 'r--', ...
    'LineWidth', 1.5);

yline(1, 'k:', 'LineWidth', 1);

xlabel('Candidate Timing Index d', ...
    'FontSize', 12);

ylabel('Schmidl-Cox Timing Metric M(d)', ...
    'FontSize', 12);

title('Noiseless Schmidl-Cox Timing Metric: Plateau Effect', ...
    'FontSize', 14);

xlim([D-10, trueUsefulStart+15]);
ylim([0.5 1.1]);

grid on;
box on;

legend('Timing Metric', ...
    'Start of CP Plateau', ...
    'True Useful Symbol Start', ...
    'Ideal Metric Level', ...
    'Location', 'best');

hold off;

%% 12. Save Figures and Results
projectRoot = fileparts(fileparts(mfilename('fullpath')));

figureDir = fullfile(projectRoot, 'results', 'figures');
dataDir = fullfile(projectRoot, 'results', 'data');

if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Save full-range timing metric
figure(1);

exportgraphics(gcf, ...
    fullfile(figureDir, 'sc_noiseless_timing_metric_full.png'), ...
    'Resolution', 300);

% Save zoomed timing metric
figure(2);

exportgraphics(gcf, ...
    fullfile(figureDir, 'sc_noiseless_timing_metric_zoom.png'), ...
    'Resolution', 300);

% Save numerical results
save(fullfile(dataDir, 'sc_noiseless_timing_metric.mat'), ...
    'candidateIndex', ...
    'P', 'R', 'M', ...
    'Nfft', 'Ncp', 'L', 'D', ...
    'trueUsefulStart', ...
    'plateauStart', 'plateauEnd', ...
    'repetitionError');

fprintf('\nNoiseless Timing Metric figures and data saved.\n');