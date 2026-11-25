%% EXP08 End-to-End Noiseless BER Validation
% Actual timing estimates drive OFDM data demodulation.
%
% Case 1: Perfect timing + known physical channel
% Case 2: Estimated timing + known physical channel
% Case 3: Estimated timing + oracle phase compensation

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2047;
rng(randomSeed);

%% 1. Common configuration

N = cfg.Nfft;
Ncp = cfg.Ncp;

symbolLength = N+Ncp;
bitsPerOFDM = N*cfg.bitsPerSymbol;

% Force subcarrier indices to be a column vector
subcarrierIndex = (0:N-1).';
subcarrierIndex = subcarrierIndex(:);

assert(isequal(size(subcarrierIndex),[N,1]));

arrivalRange = [20 80];

searchRange = [ ...
    arrivalRange(1)+1, ...
    arrivalRange(2)+2*Ncp+1];

delayList = [20 30 50 70 80];

algorithmNames = {'SC','Park','Minn','CAZAC'};

channelNames = {'Noiseless','Fixed_Multipath'};

channelResponses = {1,cfg.hMultipath(:)};

numAlgorithms = numel(algorithmNames);
numChannels = numel(channelNames);
numDelays = numel(delayList);

maxPathDelays = [0,numel(cfg.hMultipath)-1];

cpSafeMin = -Ncp+maxPathDelays;

bufferLength = arrivalRange(2) ...
    +2*symbolLength+N+max(maxPathDelays);

assert(Ncp>=max(maxPathDelays));

%% 2. Initialize results

allTimingErrors = NaN( ...
    numDelays,numChannels,numAlgorithms);

allEstimatedStarts = NaN(size(allTimingErrors));

allValid = false(size(allTimingErrors));

allTimingValid = false(size(allTimingErrors));

allCPsafe = false(size(allTimingErrors));

% Fourth dimension:
% 1 = Perfect timing
% 2 = Estimated timing
% 3 = Estimated timing + oracle compensation

bitErrors = NaN(numDelays,numChannels,numAlgorithms,3);

allBER = NaN(size(bitErrors));

allReceiverValid = false(size(bitErrors));

%% 3. Display configuration

fprintf('EXP08 End-to-End Noiseless\n');

fprintf('Random Seed = %d\n',randomSeed);

fprintf('FFT Size = %d\n',N);

fprintf('CP Length = %d\n',Ncp);

fprintf('Bits per OFDM Symbol = %d\n',bitsPerOFDM);

fprintf('Common Search Range = [%d, %d]\n', ...
    searchRange);

fprintf('Buffer Length = %d\n',bufferLength);

fprintf('Subcarrier Index Size = %d x %d\n\n', ...
    size(subcarrierIndex,1), ...
    size(subcarrierIndex,2));

fprintf(['%-18s %-8s %5s %8s %7s ' ...
    '%10s %10s %10s\n'], ...
    'Channel','Method','Delay','Error','CPsafe', ...
    'PerfectBER','EstBER','OracleBER');

%% 4. End-to-end validation

for j = 1:numDelays

    D = delayList(j);

    %% Common random data bits

    txBits = randi([0 1],bitsPerOFDM,1);

    txBits = txBits(:);

    X = qpsk_mod(txBits);

    X = X(:);

    assert(isequal(size(X),[N,1]), ...
        'QPSK symbol dimension mismatch.');

    dataSymbol = ofdm_modulate(X,cfg);

    dataSymbol = dataSymbol(:);

    dataSymbol = normalize_training_energy( ...
        dataSymbol,cfg.trainingEnergy);

    assert(numel(dataSymbol)==symbolLength);

    assert(abs(sum(abs(dataSymbol).^2) ...
        -cfg.trainingEnergy)<1e-10);

    %% Algorithm-specific training

    trainingList = { ...
        schmidl_cox_training(cfg), ...
        park_training_original(cfg), ...
        minn_training(cfg), ...
        cazac_training(cfg)};

    trueTrainingStart = D+Ncp+1;

    trueDataStart = ...
        trueTrainingStart+symbolLength;

    for a = 1:numAlgorithms

        training = trainingList{a};

        training = training(:);

        assert(numel(training)==symbolLength);

        assert(abs(sum(abs(training).^2) ...
            -cfg.trainingEnergy)<1e-10);

        %% Construct buffered transmitted frame

        txFrame = [training;dataSymbol];

        txBuffered = zeros(bufferLength,1);

        txBuffered(D+1:D+numel(txFrame)) = ...
            txFrame;

        for c = 1:numChannels

            %% Channel processing

            h = channelResponses{c};

            h = h(:);

            rxFull = conv(txBuffered,h);

            rx = rxFull(1:bufferLength);

            rx = rx(:);

            assert(numel(rx)==bufferLength);

            assert(all(isfinite(rx)));

            %% Known physical channel response
            % Force a 64-by-1 frequency-domain vector.

            if c==1

                % Unit-gain single-path channel
                H = ones(N,1);

            else

                % Fixed multipath channel
                H = fft(h(:),N);

            end

            H = H(:);

            assert(isequal(size(H),[N,1]), ...
                'Physical channel dimension mismatch.');

            assert(all(isfinite(H)), ...
                'Physical channel contains NaN or Inf.');

            assert(all(abs(H)>1e-12), ...
                'Physical channel contains spectral nulls.');

            %% Case 1: Perfect timing baseline

            [bitsPerfect,validPerfect] = ...
                ofdm_receive_at_start( ...
                rx,trueDataStart,cfg,H);

            allReceiverValid(j,c,a,1) = ...
                validPerfect;

            assert(validPerfect, ...
                'Perfect timing receiver failed.');

            bitsPerfect = bitsPerfect(:);

            assert(isequal(size(bitsPerfect), ...
                size(txBits)), ...
                'Perfect timing bit dimension mismatch.');

            nErrPerfect = ...
                sum(bitsPerfect~=txBits);

            bitErrors(j,c,a,1) = nErrPerfect;

            assert(nErrPerfect==0, ...
                'Noiseless perfect timing failed.');

            %% Actual timing synchronization

            [estTrainingStart,~,validTiming] = ...
                timing_estimator_unified( ...
                rx,cfg,algorithmNames{a}, ...
                searchRange);

            allTimingValid(j,c,a) = validTiming;

            if ~validTiming

                fprintf('%-18s %-8s %5d INVALID\n', ...
                    channelNames{c}, ...
                    algorithmNames{a},D);

                continue;

            end

            assert(isscalar(estTrainingStart));

            assert(isfinite(estTrainingStart));

            assert(estTrainingStart== ...
                round(estTrainingStart));

            assert(estTrainingStart>=searchRange(1));

            assert(estTrainingStart<=searchRange(2));

            allEstimatedStarts(j,c,a) = ...
                estTrainingStart;

            %% Estimated data FFT start

            estDataStart = ...
                estTrainingStart+symbolLength;

            % True timing error:
            % Evaluation and oracle compensation ONLY.

            timingError = ...
                estTrainingStart-trueTrainingStart;

            assert(isscalar(timingError));

            assert(isfinite(timingError));

            assert(estDataStart-trueDataStart ...
                ==timingError);

            allTimingErrors(j,c,a) = timingError;

            cpSafe = ...
                timingError>=cpSafeMin(c) ...
                && timingError<=0;

            allCPsafe(j,c,a) = cpSafe;

            %% Case 2: Actual timing, no phase compensation

            [bitsEstimated,validEstimated] = ...
                ofdm_receive_at_start( ...
                rx,estDataStart,cfg,H);

            allReceiverValid(j,c,a,2) = ...
                validEstimated;

            if ~validEstimated

                fprintf(['%-18s %-8s %5d ' ...
                    'ESTIMATED RECEIVER INVALID\n'], ...
                    channelNames{c}, ...
                    algorithmNames{a},D);

                continue;

            end

            bitsEstimated = bitsEstimated(:);

            assert(isequal(size(bitsEstimated), ...
                size(txBits)), ...
                'Estimated receiver bit dimension mismatch.');

            nErrEstimated = ...
                sum(bitsEstimated~=txBits);

            bitErrors(j,c,a,2) = nErrEstimated;

            %% Case 3: Oracle phase compensation
            %
            % True timing error is explicitly used here.
            % This is an ideal reference, not a practical
            % receiver-side timing error estimate.

            phaseRotation = exp( ...
                1j*2*pi*subcarrierIndex(:) ...
                *timingError/N);

            phaseRotation = phaseRotation(:);

            assert(isequal(size(phaseRotation),[N,1]), ...
                'Phase rotation dimension mismatch.');

            assert(all(isfinite(phaseRotation)));

            % Both operands MUST be N-by-1 vectors.
            % This prevents implicit expansion into N-by-N.

            Heffective = H(:).*phaseRotation(:);

            Heffective = Heffective(:);

            assert(isequal(size(Heffective),[N,1]), ...
                'Effective channel dimension mismatch.');

            assert(all(isfinite(Heffective)), ...
                'Effective channel contains NaN or Inf.');

            [bitsOracle,validOracle] = ...
                ofdm_receive_at_start( ...
                rx,estDataStart,cfg,Heffective);

            allReceiverValid(j,c,a,3) = ...
                validOracle;

            if ~validOracle

                fprintf(['%-18s %-8s %5d ' ...
                    'ORACLE RECEIVER INVALID\n'], ...
                    channelNames{c}, ...
                    algorithmNames{a},D);

                continue;

            end

            bitsOracle = bitsOracle(:);

            assert(isequal(size(bitsOracle), ...
                size(txBits)), ...
                'Oracle receiver bit dimension mismatch.');

            nErrOracle = ...
                sum(bitsOracle~=txBits);

            bitErrors(j,c,a,3) = nErrOracle;

            allValid(j,c,a) = true;

            %% Structural acceptance checks

            if timingError==0

                assert(nErrEstimated==0, ...
                    'Exact timing estimated BER is nonzero.');

                assert(nErrOracle==0, ...
                    'Exact timing oracle BER is nonzero.');

                assert(isequal( ...
                    bitsEstimated,bitsPerfect));

            end

            if cpSafe

                assert(nErrOracle==0, ...
                    'CP-safe oracle compensation failed.');

            end

            %% Print individual result

            fprintf(['%-18s %-8s %5d %+8d %7d ' ...
                '%10.6f %10.6f %10.6f\n'], ...
                channelNames{c}, ...
                algorithmNames{a}, ...
                D,timingError,cpSafe, ...
                nErrPerfect/bitsPerOFDM, ...
                nErrEstimated/bitsPerOFDM, ...
                nErrOracle/bitsPerOFDM);

        end

    end

end

%% 5. Calculate BER

allBER = bitErrors./bitsPerOFDM;

assert(all(bitErrors(:)>=0 ...
    | isnan(bitErrors(:))));

assert(all(bitErrors(:)<=bitsPerOFDM ...
    | isnan(bitErrors(:))));

assert(all(isfinite( ...
    bitErrors(allReceiverValid))));

assert(all(isnan( ...
    bitErrors(~allReceiverValid))));

assert(all(isfinite( ...
    allTimingErrors(allTimingValid))));

assert(all(isnan( ...
    allTimingErrors(~allTimingValid))));

assert(all(allBER(:)>=0 ...
    | isnan(allBER(:))));

assert(all(allBER(:)<=1 ...
    | isnan(allBER(:))));

assert(isequal(allValid, ...
    allTimingValid & all(allReceiverValid,4)));

%% 6. Aggregate results

fprintf('\n===== End-to-End Summary =====\n');

fprintf(['%-18s %-8s %7s %8s ' ...
    '%12s %12s %12s\n'], ...
    'Channel','Method','Valid','CPsafe', ...
    'PerfectBER','EstBER','OracleBER');

for c = 1:numChannels

    for a = 1:numAlgorithms

        validCount = sum(allValid(:,c,a));

        safeCount = sum(allCPsafe(:,c,a) ...
            & allTimingValid(:,c,a));

        berPerfect = mean( ...
            allBER(:,c,a,1),'omitnan');

        berEstimated = mean( ...
            allBER(:,c,a,2),'omitnan');

        berOracle = mean( ...
            allBER(:,c,a,3),'omitnan');

        fprintf(['%-18s %-8s %3d/%-3d %3d/%-4d ' ...
            '%12.8f %12.8f %12.8f\n'], ...
            channelNames{c}, ...
            algorithmNames{a}, ...
            validCount,numDelays, ...
            safeCount,numDelays, ...
            berPerfect,berEstimated,berOracle);

    end

end

%% 7. Save validation data

dataDir = fullfile(projectRoot,'results','data');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

save(fullfile(dataDir, ...
    'exp08_end_to_end_noiseless.mat'), ...
    'cfg','randomSeed','delayList', ...
    'algorithmNames','channelNames', ...
    'channelResponses','searchRange', ...
    'bufferLength','bitsPerOFDM','cpSafeMin', ...
    'subcarrierIndex','allEstimatedStarts', ...
    'allTimingErrors','allValid', ...
    'allTimingValid','allReceiverValid', ...
    'allCPsafe','bitErrors','allBER');

%% 8. Final validation report

fprintf('\nEXP08 Validation Report\n');

fprintf('Total Timing Estimates = %d\n', numel(allTimingErrors));

fprintf('Valid Timing Estimates = %d\n', sum(allTimingValid(:)));

fprintf('Invalid Timing Estimates = %d\n', sum(~allTimingValid(:)));

fprintf('Valid End-to-End Cases = %d\n', sum(allValid(:)));

fprintf('Invalid End-to-End Cases = %d\n', sum(~allValid(:)));

fprintf('Total Perfect Receiver Cases = %d\n', numel(allReceiverValid(:,:,:,1)));

fprintf('Valid Perfect Receiver Cases = %d\n', sum(allReceiverValid(:,:,:,1),'all'));

fprintf('Valid Estimated Receiver Cases = %d\n', sum(allReceiverValid(:,:,:,2),'all'));

fprintf('Valid Oracle Receiver Cases = %d\n', sum(allReceiverValid(:,:,:,3),'all'));

fprintf('All channel dimension checks passed.\n');

fprintf('All perfect-timing checks passed.\n');

fprintf('All valid CP-safe oracle checks passed.\n');

fprintf('All statistical consistency checks passed.\n');

fprintf('Validation data saved.\n');

fprintf('\nEXP08 end-to-end noiseless completed.\n');
