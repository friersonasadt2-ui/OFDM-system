%% EXP05 - Minn AWGN Smoke Test
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

% Fixed Minn training symbol
[training,~,~] = minn_training(cfg);

for s = 1:numSNR

    snrDb = snrList(s);
    errors = NaN(numTrials,1);
    validFlags = false(numTrials,1);

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

        txFrame = [training;dataSymbol];

        rxClean = ...
            [zeros(D,1);txFrame;zeros(N,1)];

        % 3. AWGN based on active frame power
        signalPower = mean(abs(txFrame).^2);

        noisePower = ...
            signalPower/10^(snrDb/10);

        noise = sqrt(noisePower/2)* ...
            (randn(size(rxClean)) + ...
             1j*randn(size(rxClean)));

        rx = rxClean+noise;

        % 4. Timing estimation
        [estimatedStart,~,isValid,~] = ...
            minn_estimator(rx,cfg);

        validFlags(t) = isValid;

        if isValid
            errors(t) = ...
                estimatedStart-trueStart;
        end

    end

    % 5. Calculate statistics
    allErrors(:,s) = errors;

    validErrors = errors(isfinite(errors));

    exactProb(s) = mean(errors==0);

    cpSafeProb(s) = ...
        mean(errors>=-Ncp & errors<=0);

    failProb(s) = mean(~validFlags);

    if ~isempty(validErrors)
        condMSE(s) = mean(validErrors.^2);
        meanError(s) = mean(validErrors);
    end

end

%% 6. Print results
fprintf('\nEXP05 Minn AWGN Smoke Test\n');
fprintf('Trials per SNR = %d\n',numTrials);
fprintf('Random Seed = 2027\n');

fprintf('\n');
fprintf('SNR(dB)  Exact   CP-safe  Fail    Cond.MSE   MeanError\n');

for s = 1:numSNR

    fprintf('%3d      %.4f  %.4f   %.4f  %10.4f  %+9.4f\n', ...
        snrList(s),exactProb(s), ...
        cpSafeProb(s),failProb(s), ...
        condMSE(s),meanError(s));

end

%% 7. Print timing error distributions
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

disp('EXP05 Minn AWGN smoke test completed.');
