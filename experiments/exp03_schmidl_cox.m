%% Experiment 03A: Schmidl-Cox Training Symbol Generation

clear;
clc;
close all;

%% 1. Parameters
Nfft = 64;
Ncp = 16;
L = Nfft / 2;

rng(42);

%% 2. Generate QPSK Symbols
qpskData = randi([0 3], L, 1);

qpskSymbols = pskmod(qpskData, 4, pi/4);

%% 3. Construct Frequency-Domain Training Symbol
X = zeros(Nfft, 1);

% MATLAB indices 1,3,5,... correspond to k=0,2,4,...
X(1:2:end) = qpskSymbols;

%% 4. IFFT
x = ifft(X, Nfft);

%% 5. Verify Repeated Halves
A = x(1:L);
B = x(L+1:end);

difference = max(abs(A - B));

fprintf('Maximum difference between A and B: %.3e\n', difference);

%% 6. Add Cyclic Prefix
cp = x(end-Ncp+1:end);

trainingSymbol = [cp; x];

fprintf('Training symbol length: %d samples\n', length(trainingSymbol));

%% 7. Basic Verification
assert(difference < 1e-12, ...
    'Training symbol halves are not identical.');

assert(length(trainingSymbol) == Nfft + Ncp, ...
    'Training symbol length is incorrect.');

disp('Experiment 03A basic verification PASSED.');

%% 8. Experiment 03B: Transmitted Signal Construction
% Unknown arrival delay
D = 50;

%% 9. Generate One Normal OFDM Data Symbol
dataIndices = randi([0 3], Nfft, 1);

dataSymbols = pskmod(dataIndices, 4, pi/4);

dataTime = ifft(dataSymbols, Nfft);

dataCP = dataTime(end-Ncp+1:end);

dataOFDM = [dataCP; dataTime];

%% 10. Construct Transmitted Frame
txFrame = [trainingSymbol; dataOFDM];

%% 11. Add Arrival Delay
txDelayed = [zeros(D,1); txFrame];

%% 12. Define True Timing Positions
trueCPStart = D + 1;

trueUsefulStart = D + Ncp + 1;

fprintf('\nExperiment 03B\n');

fprintf('Arrival delay: %d samples\n', D);

fprintf('True CP start: %d\n', trueCPStart);

fprintf('True useful symbol start: %d\n', trueUsefulStart);

fprintf('Total received buffer length: %d samples\n', ...
    length(txDelayed));

%% 13. Verification
assert(length(txDelayed) == D + 2*(Nfft+Ncp), ...
    'Received buffer length is incorrect.');

assert(all(txDelayed(1:D) == 0), ...
    'Delay region should contain zeros.');

assert(norm(txDelayed(D+1:D+80) - trainingSymbol) < 1e-12, ...
    'Training symbol position is incorrect.');

disp('Experiment 03B verification PASSED.');

%% 14. Experiment 03C: Schmidl-Cox Timing Metric
% Received signal (no AWGN for now)
rxSignal = txDelayed;

% Number of valid candidate positions
numCandidates = length(rxSignal) - 2*L + 1;

% Initialize timing metric arrays
P = zeros(numCandidates, 1);
R = zeros(numCandidates, 1);
M = zeros(numCandidates, 1);

%% 15. Calculate Correlation and Energy
for d = 1:numCandidates

    % First half-window
    r1 = rxSignal(d : d+L-1);

    % Second half-window
    r2 = rxSignal(d+L : d+2*L-1);

    % Delayed correlation
    P(d) = sum(conj(r1) .* r2);

    % Energy normalization
    R(d) = sum(abs(r2).^2);

    % Timing metric
    if R(d) > 1e-12
        M(d) = abs(P(d))^2 / R(d)^2;
    else
        M(d) = 0;
    end

end

%% 16. Basic Verification
fprintf('\nExperiment 03C\n');

fprintf('Number of candidate positions: %d\n', numCandidates);

fprintf('Timing metric at CP start: %.6f\n', M(trueCPStart));

fprintf('Timing metric at useful start: %.6f\n', M(trueUsefulStart));

% Verify ideal plateau
plateauValues = M(trueCPStart:trueUsefulStart);

plateauError = max(abs(plateauValues - 1));

fprintf('Maximum plateau deviation from 1: %.3e\n', plateauError);

assert(plateauError < 1e-10, ...
    'Ideal timing plateau verification failed.');

disp('Experiment 03C verification PASSED.');

%% 17. Plot Schmidl-Cox Timing Metric
figure('Color', 'w');

plot(1:numCandidates, M, ...
    'b-', 'LineWidth', 1.8);

hold on;

% True CP start
xline(trueCPStart, 'r--', ...
    'CP Start', 'LineWidth', 1.5);

% True useful symbol start
xline(trueUsefulStart, 'g--', ...
    'Useful Start', 'LineWidth', 1.5);

xlabel('Candidate Timing Position d (samples)');

ylabel('Timing Metric M(d)');

title('Experiment 03C: Schmidl-Cox Timing Metric');

grid on;

xlim([1 numCandidates]);

legend('Timing Metric', ...
    'CP Start', ...
    'Useful Start', ...
    'Location', 'best');

hold off;

%% 18. Experiment 03D: Threshold-Based Plateau Detection
threshold = 0.95;

% Find candidate positions above threshold
aboveThreshold = (M >= threshold);

% Find continuous regions
edges = diff([false; aboveThreshold; false]);

regionStarts = find(edges == 1);
regionEnds = find(edges == -1) - 1;

% Select the longest continuous region
regionLengths = regionEnds - regionStarts + 1;

[~, longestRegionIndex] = max(regionLengths);

plateauStart = regionStarts(longestRegionIndex);
plateauEnd = regionEnds(longestRegionIndex);

% Estimate timing using plateau midpoint
estimatedStart = round((plateauStart + plateauEnd) / 2);

% Timing error relative to ideal FFT window
timingError = estimatedStart - trueUsefulStart;

%% 19. Display Results
fprintf('\nExperiment 03D\n');

fprintf('Detected plateau start: %d\n', plateauStart);
fprintf('Detected plateau end: %d\n', plateauEnd);

fprintf('Estimated timing position: %d\n', estimatedStart);
fprintf('True useful start: %d\n', trueUsefulStart);

fprintf('Timing error: %d samples\n', timingError);

%% 20. CP-Safe Verification
isCPSafe = (timingError >= -Ncp) && (timingError <= 0);

fprintf('CP-safe timing: %d\n', isCPSafe);

assert(isCPSafe, ...
    'Estimated timing position is outside CP-safe range.');

disp('Experiment 03D verification PASSED.');

%% 21. Experiment 03E: AWGN Channel - Single Trial
SNR_dB = 10;

% Measure signal power over the transmitted frame
signalPower = mean(abs(txFrame).^2);

% Calculate noise power
noisePower = signalPower / (10^(SNR_dB/10));

% Generate complex AWGN
noise = sqrt(noisePower/2) * ...
    (randn(size(txDelayed)) + 1j*randn(size(txDelayed)));

% Add noise to the entire received buffer
rxNoisy = txDelayed + noise;

%% 22. Verify AWGN Power
measuredNoisePower = mean(abs(noise).^2);

measuredSNR = 10*log10(signalPower/measuredNoisePower);

fprintf('\nExperiment 03E: AWGN Verification\n');

fprintf('Target SNR: %.2f dB\n', SNR_dB);
fprintf('Measured SNR: %.2f dB\n', measuredSNR);

fprintf('Signal power: %.6e\n', signalPower);
fprintf('Noise power (target): %.6e\n', noisePower);
fprintf('Noise power (measured): %.6e\n', measuredNoisePower);

%% 23. Basic Verification
assert(abs(measuredSNR - SNR_dB) < 1.5, ...
    'Measured SNR differs too much from target.');

disp('Experiment 03E AWGN verification PASSED.');

%% 24. Experiment 03E: Noisy Timing Metric
P_noisy = zeros(numCandidates, 1);
R_noisy = zeros(numCandidates, 1);
M_noisy = zeros(numCandidates, 1);

%% 25. Calculate Noisy Correlation and Energy
for d = 1:numCandidates

    % First half-window
    r1 = rxNoisy(d : d+L-1);

    % Second half-window
    r2 = rxNoisy(d+L : d+2*L-1);

    % Delayed correlation
    P_noisy(d) = sum(conj(r1) .* r2);

    % Energy normalization
    R_noisy(d) = sum(abs(r2).^2);

    % Timing metric
    if R_noisy(d) > 1e-12
        M_noisy(d) = abs(P_noisy(d))^2 / R_noisy(d)^2;
    else
        M_noisy(d) = 0;
    end

end

%% 26. Numerical Verification
fprintf('\nExperiment 03E: Noisy Timing Metric\n');

fprintf('Clean metric at CP start: %.6f\n', ...
    M(trueCPStart));

fprintf('Noisy metric at CP start: %.6f\n', ...
    M_noisy(trueCPStart));

fprintf('Clean metric at useful start: %.6f\n', ...
    M(trueUsefulStart));

fprintf('Noisy metric at useful start: %.6f\n', ...
    M_noisy(trueUsefulStart));

fprintf('Maximum noisy timing metric: %.6f\n', ...
    max(M_noisy));

assert(all(isfinite(M_noisy)), ...
    'Noisy timing metric contains NaN or Inf.');

disp('Experiment 03E noisy metric verification PASSED.');

%% 27. Plot Clean vs Noisy Timing Metric
figure('Color', 'w');

plot(1:numCandidates, M, ...
    'b-', 'LineWidth', 1.8);

hold on;

plot(1:numCandidates, M_noisy, ...
    'm-', 'LineWidth', 1.5);

xline(trueCPStart, 'r--', ...
    'CP Start', 'LineWidth', 1.3);

xline(trueUsefulStart, 'g--', ...
    'Useful Start', 'LineWidth', 1.3);

xlabel('Candidate Timing Position d (samples)');

ylabel('Timing Metric M(d)');

title('Experiment 03E: Schmidl-Cox Timing Metric under AWGN');

legend('Noiseless', ...
    'AWGN (SNR = 10 dB)', ...
    'CP Start', ...
    'Useful Start', ...
    'Location', 'best');

grid on;

xlim([1 numCandidates]);

hold off;

%% 28. Experiment 03E: Noisy Timing Estimation
% Relative threshold factor
alpha = 0.90;

% Find maximum noisy timing metric
[maxMetric, maxIndex] = max(M_noisy);

% Calculate adaptive threshold
adaptiveThreshold = alpha * maxMetric;

%% 29. Detect Candidate Regions
aboveThresholdNoisy = (M_noisy >= adaptiveThreshold);

% Find continuous regions
edgesNoisy = diff([false; aboveThresholdNoisy; false]);

regionStartsNoisy = find(edgesNoisy == 1);
regionEndsNoisy = find(edgesNoisy == -1) - 1;

% Select the region containing the global maximum
selectedRegion = find( ...
    regionStartsNoisy <= maxIndex & ...
    regionEndsNoisy >= maxIndex, 1);

plateauStartNoisy = regionStartsNoisy(selectedRegion);
plateauEndNoisy = regionEndsNoisy(selectedRegion);

% Estimate timing using region midpoint
estimatedStartNoisy = round( ...
    (plateauStartNoisy + plateauEndNoisy) / 2);

% Calculate timing error
timingErrorNoisy = estimatedStartNoisy - trueUsefulStart;

% Evaluate timing accuracy
isExactNoisy = (timingErrorNoisy == 0);

isCPSafeNoisy = ...
    (timingErrorNoisy >= -Ncp) && ...
    (timingErrorNoisy <= 0);

%% 30. Display Results
fprintf('\nExperiment 03E: Noisy Timing Estimation\n');

fprintf('Maximum metric: %.6f\n', maxMetric);
fprintf('Maximum metric position: %d\n', maxIndex);

fprintf('Adaptive threshold: %.6f\n', adaptiveThreshold);

fprintf('Detected region: [%d, %d]\n', ...
    plateauStartNoisy, plateauEndNoisy);

fprintf('Estimated timing position: %d\n', ...
    estimatedStartNoisy);

fprintf('True useful start: %d\n', trueUsefulStart);

fprintf('Timing error: %d samples\n', timingErrorNoisy);

fprintf('Exact timing: %d\n', isExactNoisy);
fprintf('CP-safe timing: %d\n', isCPSafeNoisy);

%% 31. Basic Verification
assert(isfinite(estimatedStartNoisy), ...
    'Invalid noisy timing estimate.');

assert(estimatedStartNoisy >= 1 && ...
    estimatedStartNoisy <= numCandidates, ...
    'Estimated position is outside valid range.');

disp('Experiment 03E noisy estimation verification PASSED.');

%% 32. Verify Schmidl-Cox Metric Function
[P_func, R_func, M_func] = ...
    schmidl_cox_metric(rxNoisy, L);

% Compare with original implementation
differenceP = max(abs(P_func - P_noisy));

differenceR = max(abs(R_func - R_noisy));

differenceM = max(abs(M_func - M_noisy));

fprintf('\nSchmidl-Cox Function Verification\n');

fprintf('Maximum P difference: %.3e\n', differenceP);

fprintf('Maximum R difference: %.3e\n', differenceR);

fprintf('Maximum M difference: %.3e\n', differenceM);

assert(differenceP < 1e-12, ...
    'Correlation function mismatch.');

assert(differenceR < 1e-12, ...
    'Energy function mismatch.');

assert(differenceM < 1e-12, ...
    'Timing metric function mismatch.');

disp('Schmidl-Cox metric function verification PASSED.');

%% 33. Verify Schmidl-Cox Timing Estimator
[estimatedStartFunc, estimatorInfo] = ...
    schmidl_cox_estimator(M_noisy, alpha);

fprintf('\nSchmidl-Cox Estimator Verification\n');

fprintf('Original estimated start: %d\n', ...
    estimatedStartNoisy);

fprintf('Function estimated start: %d\n', ...
    estimatedStartFunc);

fprintf('Function plateau: [%d, %d]\n', ...
    estimatorInfo.plateauStart, ...
    estimatorInfo.plateauEnd);

fprintf('Function threshold: %.6f\n', ...
    estimatorInfo.threshold);

%% Verification
assert(estimatedStartFunc == estimatedStartNoisy, ...
    'Timing estimator mismatch.');

assert(estimatorInfo.plateauStart == plateauStartNoisy, ...
    'Plateau start mismatch.');

assert(estimatorInfo.plateauEnd == plateauEndNoisy, ...
    'Plateau end mismatch.');

assert(estimatorInfo.detected, ...
    'Timing estimator failed to detect a region.');

disp('Schmidl-Cox estimator function verification PASSED.');