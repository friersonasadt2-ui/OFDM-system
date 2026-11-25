%% EXP07 Unified Four-Algorithm Smoke Test
% Paired random scenarios for all four algorithms.
% AWGN and fixed three-path multipath channels.

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2044;
rng(randomSeed);

%% 1. Common experimental configuration

N = cfg.Nfft;
Ncp = cfg.Ncp;

arrivalRange = [20 80];

searchRange = [ ...
    arrivalRange(1)+1, ...
    arrivalRange(2)+2*Ncp+1];

snrList = [0 10 20];

numTrials = 100;

algorithmNames = {'SC','Park','Minn','CAZAC'};

channelNames = {'AWGN','Fixed_Multipath'};

channelResponses = {1,cfg.hMultipath(:)};

numAlgorithms = numel(algorithmNames);
numChannels = numel(channelNames);
numSNR = numel(snrList);

maxPathDelays = [0,numel(cfg.hMultipath)-1];

cpSafeMin = -Ncp+maxPathDelays;

bufferLength = arrivalRange(2) ...
    +2*(N+Ncp)+N+max(maxPathDelays);

%% 2. Initialize trial-level results

allErrors = NaN( ...
    numTrials,numSNR,numChannels,numAlgorithms);

allValid = false( ...
    numTrials,numSNR,numChannels,numAlgorithms);

allPeakMetrics = NaN(size(allErrors));

allSignalPowers = NaN(size(allErrors));

allArrivalDelays = zeros(numTrials,numSNR);

%% 3. Initialize summary statistics

exactProbability = zeros( ...
    numSNR,numChannels,numAlgorithms);

cpSafeProbability = zeros(size(exactProbability));

failProbability = zeros(size(exactProbability));

conditionalMSE = NaN(size(exactProbability));

meanError = NaN(size(exactProbability));

maxAbsError = NaN(size(exactProbability));

%% 4. Display configuration

fprintf('EXP07 Unified Smoke Test\n');

fprintf('Random Seed = %d\n',randomSeed);

fprintf('Trials per SNR = %d\n',numTrials);

fprintf('Arrival Range = [%d, %d]\n',arrivalRange);

fprintf('Common Search Range = [%d, %d]\n',searchRange);

fprintf('Buffer Length = %d\n',bufferLength);

fprintf('Total Estimates = %d\n\n', ...
    numTrials*numSNR*numChannels*numAlgorithms);

%% 5. Monte Carlo simulation

tic;

for s = 1:numSNR

    snrDb = snrList(s);

    for trial = 1:numTrials

        %% 5.1 Common random arrival delay

        D = randi(arrivalRange);

        allArrivalDelays(trial,s) = D;

        trueStart = D+Ncp+1;

        %% 5.2 Common QPSK data symbol

        dataIdx = randi([0 3],N,1);

        dataFreq = qammod(dataIdx,4, ...
            'UnitAveragePower',true);

        xData = ifft(dataFreq,N);

        dataSymbol = ...
            [xData(end-Ncp+1:end);xData];

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        assert(abs(sum(abs(dataSymbol).^2) ...
            -cfg.trainingEnergy)<1e-10);

        %% 5.3 Common base noise

        baseNoise = ...
            (randn(bufferLength,1) ...
            +1j*randn(bufferLength,1))/sqrt(2);

        %% 5.4 Algorithm-specific training symbols

        trainingList = { ...
            schmidl_cox_training(cfg), ...
            park_training_original(cfg), ...
            minn_training(cfg), ...
            cazac_training(cfg)};

        %% 5.5 Process four algorithms

        for a = 1:numAlgorithms

            training = trainingList{a};

            assert(numel(training)==N+Ncp);

            assert(abs(sum(abs(training).^2) ...
                -cfg.trainingEnergy)<1e-10);

            txFrame = [training;dataSymbol];

            txBuffered = zeros(bufferLength,1);

            txBuffered(D+1:D+numel(txFrame)) = ...
                txFrame;

            %% 5.6 Process both channel conditions

            for c = 1:numChannels

                h = channelResponses{c};

                rxFull = conv(txBuffered,h);

                rxClean = rxFull(1:bufferLength);

                %% Received active-frame power

                activeFrame = ...
                    rxClean(D+1:D+numel(txFrame));

                signalPower = ...
                    mean(abs(activeFrame).^2);

                assert(isfinite(signalPower));
                assert(signalPower>0);

                allSignalPowers(trial,s,c,a) = ...
                    signalPower;

                if c==1
                    assert(abs(signalPower-1)<1e-10);
                end

                %% Add complex AWGN

                noisePower = ...
                    signalPower/10^(snrDb/10);

                rx = rxClean ...
                    +sqrt(noisePower)*baseNoise;

                %% Unified timing estimation

                [est,peak,valid,info] = ...
                    timing_estimator_unified( ...
                    rx,cfg,algorithmNames{a}, ...
                    searchRange);

                %% Verify common candidate support

                candidateStarts = ...
                    info.usefulIndex(info.searchValid);

                assert(isequal(candidateStarts, ...
                    (searchRange(1):searchRange(2)).'));

                %% Save trial results

                allValid(trial,s,c,a) = valid;

                if valid

                    assert(isfinite(est));
                    assert(est==round(est));

                    assert(est>=searchRange(1));
                    assert(est<=searchRange(2));

                    allErrors(trial,s,c,a) = ...
                        est-trueStart;

                    allPeakMetrics(trial,s,c,a) = ...
                        peak;

                end

            end

        end

    end

    fprintf('SNR = %2d dB completed.\n',snrDb);

end

elapsedTime = toc;

%% 6. Calculate statistics

for s = 1:numSNR

    for c = 1:numChannels

        for a = 1:numAlgorithms

            errors = allErrors(:,s,c,a);

            validErrors = errors(isfinite(errors));

            exactProbability(s,c,a) = ...
                mean(errors==0);

            cpSafeProbability(s,c,a) = ...
                mean(errors>=cpSafeMin(c) ...
                & errors<=0);

            failProbability(s,c,a) = ...
                mean(~allValid(:,s,c,a));

            if ~isempty(validErrors)

                conditionalMSE(s,c,a) = ...
                    mean(validErrors.^2);

                meanError(s,c,a) = ...
                    mean(validErrors);

                maxAbsError(s,c,a) = ...
                    max(abs(validErrors));

            end

        end

    end

end

%% 7. Statistical consistency checks

assert(all(isfinite(allErrors(allValid))));

assert(all(isnan(allErrors(~allValid))));

assert(all(exactProbability(:) ...
    <=cpSafeProbability(:)));

assert(all(cpSafeProbability(:) ...
    <=1-failProbability(:)+1e-12));

assert(all(isfinite(allSignalPowers(:))));

assert(all(allSignalPowers(:)>0));

%% 8. Display channel-specific results

for c = 1:numChannels

    fprintf('\n%s\n',channelNames{c});

    fprintf('%5s %-8s %8s %9s %8s %12s %10s\n', ...
        'SNR','Method','Exact','CP-safe', ...
        'Fail','Cond.MSE','MeanError');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf('%5d %-8s %8.4f %9.4f %8.4f %12.4f %+10.4f\n', ...
                snrList(s), ...
                algorithmNames{a}, ...
                exactProbability(s,c,a), ...
                cpSafeProbability(s,c,a), ...
                failProbability(s,c,a), ...
                conditionalMSE(s,c,a), ...
                meanError(s,c,a));

        end

    end

end

%% 9. Error distributions

for c = 1:numChannels

    fprintf('\n===== %s Error Distributions =====\n', ...
        channelNames{c});

    for s = 1:numSNR

        fprintf('\nSNR = %d dB\n',snrList(s));

        for a = 1:numAlgorithms

            errors = allErrors(:,s,c,a);

            finiteErrors = errors(isfinite(errors));

            uniqueErrors = unique(finiteErrors);

            fprintf('%-8s | ',algorithmNames{a});

            for k = 1:numel(uniqueErrors)

                e = uniqueErrors(k);

                fprintf('%+g:%d  ', ...
                    e,sum(finiteErrors==e));

            end

            fprintf('| Fail:%d\n',sum(isnan(errors)));

        end

    end

end

%% 10. Save smoke-test results

dataDir = fullfile(projectRoot,'results','data');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

save(fullfile(dataDir,'exp07_unified_smoke.mat'), ...
    'cfg','randomSeed','arrivalRange','searchRange', ...
    'snrList','numTrials','bufferLength', ...
    'algorithmNames','channelNames','channelResponses', ...
    'cpSafeMin','allArrivalDelays', ...
    'allErrors','allValid','allPeakMetrics', ...
    'allSignalPowers','exactProbability', ...
    'cpSafeProbability','failProbability', ...
    'conditionalMSE','meanError','maxAbsError', ...
    'elapsedTime');

fprintf('\nTotal Estimates = %d\n',numel(allErrors));

fprintf('Elapsed Time = %.2f seconds\n',elapsedTime);

fprintf('All statistical consistency checks passed.\n');

fprintf('Smoke test data saved.\n');

fprintf('\nEXP07 unified smoke test completed.\n');
