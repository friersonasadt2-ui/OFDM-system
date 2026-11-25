%% EXP08 Four-Algorithm BER Smoke Test
% Case 1: Perfect timing + known physical channel
% Case 2: Estimated timing + known physical channel
% Case 3: Estimated timing + oracle phase compensation
% All algorithms share arrival delay, data bits,
% channel conditions and base Gaussian noise.

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2048;
rng(randomSeed);

%% 1. Common configuration

N = cfg.Nfft;
Ncp = cfg.Ncp;

symbolLength = N+Ncp;
bitsPerOFDM = N*cfg.bitsPerSymbol;

subcarrierIndex = (0:N-1).';
subcarrierIndex = subcarrierIndex(:);

arrivalRange = [20 80];

searchRange = [ ...
    arrivalRange(1)+1, ...
    arrivalRange(2)+2*Ncp+1];

snrList = [0 10 20];

numTrials = 100;

algorithmNames = {'SC','Park','Minn','CAZAC'};

channelNames = {'AWGN','Fixed_Multipath'};

channelResponses = {1,cfg.hMultipath(:)};

receiverNames = { ...
    'Perfect', ...
    'Estimated', ...
    'Oracle'};

numAlgorithms = numel(algorithmNames);
numChannels = numel(channelNames);
numSNR = numel(snrList);
numReceivers = numel(receiverNames);

maxPathDelays = [0,numel(cfg.hMultipath)-1];

cpSafeMin = -Ncp+maxPathDelays;

bufferLength = arrivalRange(2) ...
    +2*symbolLength+N+max(maxPathDelays);

%% 2. Prepare physical channel responses

Hlist = cell(numChannels,1);

for c = 1:numChannels

    h = channelResponses{c};
    h = h(:);

    if c==1

        H = ones(N,1);

    else

        H = fft(h,N);
        H = H(:);

    end

    assert(isequal(size(H),[N,1]));

    assert(all(isfinite(H)));

    assert(all(abs(H)>1e-12));

    Hlist{c} = H;

end

%% 3. Initialize trial-level results

timingSize = [ ...
    numTrials,numSNR, ...
    numChannels,numAlgorithms];

receiverSize = [timingSize,numReceivers];

allTimingErrors = NaN(timingSize);

allTimingValid = false(timingSize);

allCPsafe = false(timingSize);

allEstimatedStarts = NaN(timingSize);

allPeakMetrics = NaN(timingSize);

allSignalPowers = NaN(timingSize);

allArrivalDelays = zeros(numTrials,numSNR);

allBitErrors = NaN(receiverSize);

allReceiverValid = false(receiverSize);

%% 4. Display configuration

fprintf('EXP08 BER Smoke Test\n');

fprintf('Random Seed = %d\n',randomSeed);

fprintf('FFT Size = %d\n',N);

fprintf('CP Length = %d\n',Ncp);

fprintf('Bits per Frame = %d\n',bitsPerOFDM);

fprintf('Trials per SNR = %d\n',numTrials);

fprintf('Arrival Range = [%d, %d]\n',arrivalRange);

fprintf('Common Search Range = [%d, %d]\n', ...
    searchRange);

fprintf('Buffer Length = %d\n',bufferLength);

fprintf('Total Timing Estimates = %d\n\n', ...
    prod(timingSize));

%% 5. Monte Carlo simulation

tic;

for s = 1:numSNR

    snrDb = snrList(s);

    for trial = 1:numTrials

        %% 5.1 Common arrival delay

        D = randi(arrivalRange);

        allArrivalDelays(trial,s) = D;

        trueTrainingStart = D+Ncp+1;

        trueDataStart = ...
            trueTrainingStart+symbolLength;

        %% 5.2 Common random data bits

        txBits = randi([0 1],bitsPerOFDM,1);

        txBits = txBits(:);

        X = qpsk_mod(txBits);

        X = X(:);

        assert(isequal(size(X),[N,1]));

        dataSymbol = ofdm_modulate(X,cfg);

        dataSymbol = dataSymbol(:);

        dataSymbol = normalize_training_energy( ...
            dataSymbol,cfg.trainingEnergy);

        assert(numel(dataSymbol)==symbolLength);

        assert(abs(sum(abs(dataSymbol).^2) ...
            -cfg.trainingEnergy)<1e-10);

        %% 5.3 Common base noise

        baseNoise = ...
            (randn(bufferLength,1) ...
            +1j*randn(bufferLength,1))/sqrt(2);

        %% 5.4 Algorithm-specific training

        trainingList = { ...
            schmidl_cox_training(cfg), ...
            park_training_original(cfg), ...
            minn_training(cfg), ...
            cazac_training(cfg)};

        %% 5.5 Process all four algorithms

        for a = 1:numAlgorithms

            training = trainingList{a};

            training = training(:);

            assert(numel(training)==symbolLength);

            assert(abs(sum(abs(training).^2) ...
                -cfg.trainingEnergy)<1e-10);

            txFrame = [training;dataSymbol];

            txBuffered = zeros(bufferLength,1);

            txBuffered(D+1:D+numel(txFrame)) = ...
                txFrame;

            %% 5.6 Process both channel conditions

            for c = 1:numChannels

                h = channelResponses{c};
                h = h(:);

                H = Hlist{c};
                H = H(:);

                assert(isequal(size(H),[N,1]));

                rxFull = conv(txBuffered,h);

                rxClean = rxFull(1:bufferLength);

                rxClean = rxClean(:);

                %% Active received-frame power

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

                %% Add common scaled AWGN

                noisePower = ...
                    signalPower/10^(snrDb/10);

                rx = rxClean ...
                    +sqrt(noisePower)*baseNoise;

                rx = rx(:);

                %% Case 1: Perfect timing baseline

                [bitsPerfect,validPerfect] = ...
                    ofdm_receive_at_start( ...
                    rx,trueDataStart,cfg,H);

                allReceiverValid(trial,s,c,a,1) = ...
                    validPerfect;

                assert(validPerfect, ...
                    'Perfect timing receiver failed.');

                bitsPerfect = bitsPerfect(:);

                assert(isequal(size(bitsPerfect), ...
                    size(txBits)));

                allBitErrors(trial,s,c,a,1) = ...
                    sum(bitsPerfect~=txBits);

                %% Actual timing synchronization

                [estTrainingStart,peak,validTiming] = ...
                    timing_estimator_unified( ...
                    rx,cfg,algorithmNames{a}, ...
                    searchRange);

                allTimingValid(trial,s,c,a) = ...
                    validTiming;

                if ~validTiming

                    continue;

                end

                assert(isscalar(estTrainingStart));

                assert(isfinite(estTrainingStart));

                assert(estTrainingStart== ...
                    round(estTrainingStart));

                assert(estTrainingStart>=searchRange(1));

                assert(estTrainingStart<=searchRange(2));

                allEstimatedStarts(trial,s,c,a) = ...
                    estTrainingStart;

                allPeakMetrics(trial,s,c,a) = peak;

                %% Estimated useful data FFT start

                estDataStart = ...
                    estTrainingStart+symbolLength;

                % True error is used only for
                % evaluation and oracle compensation.

                timingError = ...
                    estTrainingStart-trueTrainingStart;

                assert(isscalar(timingError));

                assert(estDataStart-trueDataStart ...
                    ==timingError);

                allTimingErrors(trial,s,c,a) = ...
                    timingError;

                allCPsafe(trial,s,c,a) = ...
                    timingError>=cpSafeMin(c) ...
                    && timingError<=0;

                %% Case 2: Estimated timing

                [bitsEstimated,validEstimated] = ...
                    ofdm_receive_at_start( ...
                    rx,estDataStart,cfg,H);

                allReceiverValid(trial,s,c,a,2) = ...
                    validEstimated;

                if validEstimated

                    bitsEstimated = bitsEstimated(:);

                    assert(isequal(size(bitsEstimated), ...
                        size(txBits)));

                    allBitErrors(trial,s,c,a,2) = ...
                        sum(bitsEstimated~=txBits);

                end

                %% Case 3: Oracle phase compensation

                phaseRotation = exp( ...
                    1j*2*pi*subcarrierIndex(:) ...
                    *timingError/N);

                phaseRotation = phaseRotation(:);

                assert(isequal( ...
                    size(phaseRotation),[N,1]));

                Heffective = ...
                    H(:).*phaseRotation(:);

                Heffective = Heffective(:);

                assert(isequal( ...
                    size(Heffective),[N,1]));

                [bitsOracle,validOracle] = ...
                    ofdm_receive_at_start( ...
                    rx,estDataStart,cfg,Heffective);

                allReceiverValid(trial,s,c,a,3) = ...
                    validOracle;

                if validOracle

                    bitsOracle = bitsOracle(:);

                    assert(isequal(size(bitsOracle), ...
                        size(txBits)));

                    allBitErrors(trial,s,c,a,3) = ...
                        sum(bitsOracle~=txBits);

                end

                %% Exact timing consistency check
                % With identical timing and identical
                % received samples, all three receivers
                % must return the same decisions.

                if timingError==0

                    assert(validEstimated);

                    assert(validOracle);

                    assert(isequal( ...
                        bitsEstimated,bitsPerfect));

                    assert(isequal( ...
                        bitsOracle,bitsPerfect));

                end

            end

        end

    end

    fprintf('SNR = %2d dB completed.\n',snrDb);

end

elapsedTime = toc;

%% 6. Initialize summary statistics

summarySize = [numSNR,numChannels,numAlgorithms];

berSize = [summarySize,numReceivers];

exactProbability = zeros(summarySize);

cpSafeProbability = zeros(summarySize);

timingFailProbability = zeros(summarySize);

conditionalBER = NaN(berSize);

frameFailProbability = zeros(berSize);

totalBitErrors = zeros(berSize);

decodedBits = zeros(berSize);

lostBits = zeros(berSize);

bitDeliveryFailureRatio = zeros(berSize);

offeredBits = numTrials*bitsPerOFDM;

%% 7. Calculate statistics

for s = 1:numSNR

    for c = 1:numChannels

        for a = 1:numAlgorithms

            %% Timing statistics

            errors = allTimingErrors(:,s,c,a);

            exactProbability(s,c,a) = ...
                mean(errors==0);

            cpSafeProbability(s,c,a) = ...
                mean(allCPsafe(:,s,c,a));

            timingFailProbability(s,c,a) = ...
                mean(~allTimingValid(:,s,c,a));

            %% BER receiver statistics

            for r = 1:numReceivers

                validMask = ...
                    allReceiverValid(:,s,c,a,r);

                trialErrors = ...
                    allBitErrors(:,s,c,a,r);

                numValidFrames = sum(validMask);

                numFailedFrames = ...
                    numTrials-numValidFrames;

                nErrors = sum(trialErrors(validMask));

                nDecodedBits = ...
                    numValidFrames*bitsPerOFDM;

                nLostBits = ...
                    numFailedFrames*bitsPerOFDM;

                totalBitErrors(s,c,a,r) = nErrors;

                decodedBits(s,c,a,r) = nDecodedBits;

                lostBits(s,c,a,r) = nLostBits;

                frameFailProbability(s,c,a,r) = ...
                    numFailedFrames/numTrials;

                if nDecodedBits>0

                    conditionalBER(s,c,a,r) = ...
                        nErrors/nDecodedBits;

                end

                % Lost bits are NOT ordinary bit errors.
                % This separate metric counts all lost
                % and incorrectly decoded bits.

                bitDeliveryFailureRatio(s,c,a,r) = ...
                    (nErrors+nLostBits)/offeredBits;

            end

        end

    end

end

%% 8. Statistical consistency checks

assert(all(isfinite( ...
    allTimingErrors(allTimingValid))));

assert(all(isnan( ...
    allTimingErrors(~allTimingValid))));

assert(all(isfinite( ...
    allBitErrors(allReceiverValid))));

assert(all(isnan( ...
    allBitErrors(~allReceiverValid))));

assert(all(exactProbability(:) ...
    <=cpSafeProbability(:)));

assert(all(cpSafeProbability(:) ...
    <=1-timingFailProbability(:)+1e-12));

assert(all(totalBitErrors(:)>=0));

assert(all(totalBitErrors(:)<=decodedBits(:)));

assert(all(decodedBits(:)+lostBits(:) ...
    ==offeredBits));

assert(all(bitDeliveryFailureRatio(:)>=0 ...
    & bitDeliveryFailureRatio(:)<=1));

finiteBER = ...
    conditionalBER(isfinite(conditionalBER));

assert(all(finiteBER>=0 & finiteBER<=1));

assert(all(isfinite(allSignalPowers(:))));

assert(all(allSignalPowers(:)>0));

assert(all(allReceiverValid(:,:,:,:,1),'all'));

%% 9. Display timing results

for c = 1:numChannels

    fprintf('\n%s Timing Results\n', ...
        channelNames{c});

    fprintf('%5s %-8s %10s %10s %10s\n', ...
        'SNR','Method','Exact','CP-safe','Fail');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf('%5d %-8s %10.4f %10.4f %10.4f\n', ...
                snrList(s), ...
                algorithmNames{a}, ...
                exactProbability(s,c,a), ...
                cpSafeProbability(s,c,a), ...
                timingFailProbability(s,c,a));

        end

    end

end

%% 10. Display BER results

for c = 1:numChannels

    fprintf('\n%s BER Results\n', ...
        channelNames{c});

    fprintf(['%5s %-8s %12s %12s %12s ' ...
        '%10s\n'], ...
        'SNR','Method','PerfectBER', ...
        'EstBER','OracleBER','EstFail');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf(['%5d %-8s %12.6e %12.6e ' ...
                '%12.6e %10.4f\n'], ...
                snrList(s), ...
                algorithmNames{a}, ...
                conditionalBER(s,c,a,1), ...
                conditionalBER(s,c,a,2), ...
                conditionalBER(s,c,a,3), ...
                frameFailProbability(s,c,a,2));

        end

    end

end

%% 11. Display actual bit-error counts

for c = 1:numChannels

    fprintf('\n%s Bit Error Counts\n', ...
        channelNames{c});

    fprintf('%5s %-8s %10s %10s %10s\n', ...
        'SNR','Method','Perfect','Estimated','Oracle');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf('%5d %-8s %10d %10d %10d\n', ...
                snrList(s), ...
                algorithmNames{a}, ...
                totalBitErrors(s,c,a,1), ...
                totalBitErrors(s,c,a,2), ...
                totalBitErrors(s,c,a,3));

        end

    end

end

%% 12. Save experiment data

dataDir = fullfile(projectRoot,'results','data');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

save(fullfile(dataDir,'exp08_ber_smoke.mat'), ...
    'cfg','randomSeed', ...
    'arrivalRange','searchRange', ...
    'snrList','numTrials', ...
    'algorithmNames','channelNames', ...
    'receiverNames','channelResponses', ...
    'Hlist','cpSafeMin','bufferLength', ...
    'bitsPerOFDM','offeredBits', ...
    'allArrivalDelays','allTimingErrors', ...
    'allTimingValid','allCPsafe', ...
    'allEstimatedStarts','allPeakMetrics', ...
    'allSignalPowers','allBitErrors', ...
    'allReceiverValid','exactProbability', ...
    'cpSafeProbability','timingFailProbability', ...
    'conditionalBER','frameFailProbability', ...
    'totalBitErrors','decodedBits','lostBits', ...
    'bitDeliveryFailureRatio','elapsedTime');

%% 13. Final validation report

fprintf('\nEXP08 Smoke Test Report\n');

fprintf('Total Timing Estimates = %d\n', ...
    numel(allTimingErrors));

fprintf('Valid Timing Estimates = %d\n', ...
    sum(allTimingValid(:)));

fprintf('Invalid Timing Estimates = %d\n', ...
    sum(~allTimingValid(:)));

fprintf('Valid Perfect Receivers = %d\n', ...
    sum(allReceiverValid(:,:,:,:,1),'all'));

fprintf('Valid Estimated Receivers = %d\n', ...
    sum(allReceiverValid(:,:,:,:,2),'all'));

fprintf('Valid Oracle Receivers = %d\n', ...
    sum(allReceiverValid(:,:,:,:,3),'all'));

fprintf('Offered Bits per SNR/Channel/Method = %d\n', ...
    offeredBits);

fprintf('Elapsed Time = %.2f seconds\n',elapsedTime);

fprintf('All statistical consistency checks passed.\n');

fprintf('BER smoke test data saved.\n');

fprintf('\nEXP08 BER smoke test completed.\n');
