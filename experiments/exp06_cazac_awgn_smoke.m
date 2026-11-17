%% EXP06 - CAZAC AWGN Smoke Test
clear; clc;

cfg = default_config();
rng(2027);

N = cfg.Nfft;
Ncp = cfg.Ncp;

snrList = [0 10 20];
numTrials = 100;
numSNR = numel(snrList);

exactProb = zeros(numSNR,1);
cpSafeProb = zeros(numSNR,1);
failProb = zeros(numSNR,1);
condMSE = NaN(numSNR,1);
meanError = NaN(numSNR,1);

allErrors = NaN(numTrials,numSNR);

% Fixed CAZAC training
[training,~,~] = cazac_training(cfg);

assert(numel(training)==N+Ncp);
assert(abs(sum(abs(training).^2)-80)<1e-10);

%% Monte Carlo

for s = 1:numSNR

    snrDb = snrList(s);
    errors = NaN(numTrials,1);

    for t = 1:numTrials

        % 1. Random QPSK OFDM data
        dataIndices = randi([0 3],N,1);
        dataQPSK = pskmod(dataIndices,4,pi/4);

        dataTime = ifft(dataQPSK,N);
        dataSymbol = ...
            [dataTime(end-Ncp+1:end);dataTime];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        % 2. Random arrival delay
        D = randi([20 80]);
        trueStart = D+Ncp+1;

        % 3. Transmitted frame
        txFrame = [training;dataSymbol];

        rxClean = ...
            [zeros(D,1);txFrame;zeros(N,1)];

        % 4. AWGN based on active-frame power
        signalPower = mean(abs(txFrame).^2);

        noisePower = signalPower/10^(snrDb/10);

        noise = sqrt(noisePower/2)* ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean+noise;

        % 5. CAZAC timing estimation
        [estimatedStart,~,isValid,peakIndex] = ...
            cazac_estimator(rx,cfg);

        if isValid && isfinite(estimatedStart)

            assert(estimatedStart==peakIndex+Ncp, ...
                'Training-to-FFT index conversion failed.');

            errors(t) = estimatedStart-trueStart;

        end

    end

    %% Statistics
    allErrors(:,s) = errors;

    validErrors = errors(isfinite(errors));

    exactProb(s) = mean(errors==0);

    cpSafeProb(s) = ...
        mean(errors>=-Ncp & errors<=0);

    failProb(s) = mean(~isfinite(errors));

    if ~isempty(validErrors)
        condMSE(s) = mean(validErrors.^2);
        meanError(s) = mean(validErrors);
    end

end

%% Print results
fprintf('\nEXP06 CAZAC AWGN Smoke Test\n');
fprintf('Trials per SNR = %d\n',numTrials);
fprintf('Random Seed = 2027\n');

fprintf('\n');
fprintf('SNR(dB)  Exact   CP-safe  Fail    Cond.MSE   MeanError\n');

for s = 1:numSNR

    fprintf('%3d      %.4f  %.4f   %.4f  %10.4f  %+9.4f\n', ...
        snrList(s),exactProb(s),cpSafeProb(s), ...
        failProb(s),condMSE(s),meanError(s));

end

%% Error distributions
for s = 1:numSNR

    errors = allErrors(:,s);
    validErrors = errors(isfinite(errors));

    fprintf('\nSNR = %d dB\n',snrList(s));

    if ~isempty(validErrors)

        [uniqueErrors,~,idx] = unique(validErrors);
        counts = accumarray(idx,1);

        for k = 1:numel(uniqueErrors)

            fprintf('Error %+d : %d trials\n', ...
                uniqueErrors(k),counts(k));

        end

    end

    fprintf('No Estimate : %d trials\n', ...
        sum(~isfinite(errors)));

end

disp('EXP06 CAZAC AWGN smoke test completed.');