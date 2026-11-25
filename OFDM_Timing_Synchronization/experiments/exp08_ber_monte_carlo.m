%% EXP08 Four-Algorithm Formal BER Monte Carlo
% Case 1: Perfect timing + known physical channel
% Case 2: Estimated timing + known physical channel
% Case 3: Estimated timing + oracle phase compensation
clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

randomSeed = 2049;
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

snrList = cfg.SNR_dB(:);

numTrials = cfg.numTrials;

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

fprintf('EXP08 Formal BER Monte Carlo\n');

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


%% 9. Additional formal experiment checks
assert(numel(allTimingErrors)== ...
    numTrials*numSNR*numChannels*numAlgorithms);

assert(all(allArrivalDelays(:)>=arrivalRange(1)));

assert(all(allArrivalDelays(:)<=arrivalRange(2)));

assert(all(abs(allSignalPowers(:,:,1,:)-1) ...
    <1e-10,'all'));

validStarts = ...
    allEstimatedStarts(allTimingValid);

assert(all(validStarts>=searchRange(1)));

assert(all(validStarts<=searchRange(2)));

assert(all(validStarts==round(validStarts)));

trueStarts = allArrivalDelays+Ncp+1;

reconstructedErrors = allEstimatedStarts ...
    -reshape(trueStarts,numTrials,numSNR,1,1);

assert(isequaln(reconstructedErrors, ...
    allTimingErrors));

% Verify frame accounting

assert(all(decodedBits(:)+lostBits(:) ...
    ==offeredBits));

assert(all(totalBitErrors(:)<=decodedBits(:)));

assert(all(abs(frameFailProbability(:) ...
    -lostBits(:)/offeredBits)<1e-12));

assert(all(abs(bitDeliveryFailureRatio(:) ...
    -(totalBitErrors(:)+lostBits(:)) ...
    /offeredBits)<1e-12));

% Recalculate BER for all groups with decoded bits

nonemptyMask = decodedBits>0;

recalculatedBER = ...
    totalBitErrors(nonemptyMask) ...
    ./decodedBits(nonemptyMask);

assert(all(abs( ...
    conditionalBER(nonemptyMask) ...
    -recalculatedBER)<1e-12));

fprintf('\nAll formal BER consistency checks passed.\n');

%% 10. Additional timing statistics

conditionalMSE = NaN(summarySize);

meanTimingError = NaN(summarySize);

for s = 1:numSNR

    for c = 1:numChannels

        for a = 1:numAlgorithms

            errors = allTimingErrors(:,s,c,a);

            validErrors = ...
                errors(isfinite(errors));

            if ~isempty(validErrors)

                conditionalMSE(s,c,a) = ...
                    mean(validErrors.^2);

                meanTimingError(s,c,a) = ...
                    mean(validErrors);

            end

        end

    end

end

%% 11. CP-safe and unsafe BER decomposition
% These statistics help explain which residual timing
% errors can be handled by oracle phase compensation.
% Use the receiver validity mask in all calculations.

safeBER = NaN(berSize);

unsafeBER = NaN(berSize);

safeFrameCount = zeros(berSize);

unsafeFrameCount = zeros(berSize);

for s = 1:numSNR

    for c = 1:numChannels

        for a = 1:numAlgorithms

            timingValid = ...
                allTimingValid(:,s,c,a);

            timingSafe = ...
                allCPsafe(:,s,c,a);

            for r = 1:numReceivers

                valid = ...
                    allReceiverValid(:,s,c,a,r);

                errors = ...
                    allBitErrors(:,s,c,a,r);

                safeMask = ...
                    timingValid & timingSafe & valid;

                unsafeMask = ...
                    timingValid & ~timingSafe & valid;

                nSafe = sum(safeMask);

                nUnsafe = sum(unsafeMask);

                safeFrameCount(s,c,a,r) = nSafe;

                unsafeFrameCount(s,c,a,r) = nUnsafe;

                if nSafe>0

                    safeBER(s,c,a,r) = ...
                        sum(errors(safeMask)) ...
                        /(nSafe*bitsPerOFDM);

                end

                if nUnsafe>0

                    unsafeBER(s,c,a,r) = ...
                        sum(errors(unsafeMask)) ...
                        /(nUnsafe*bitsPerOFDM);

                end

            end

        end

    end

end

%% 12. Construct BER summary table

[S,C,A,R] = ndgrid( ...
    1:numSNR, ...
    1:numChannels, ...
    1:numAlgorithms, ...
    1:numReceivers);

snrColumn = reshape( ...
    snrList(S(:)),[],1);

channelColumn = reshape( ...
    string(channelNames(C(:))),[],1);

algorithmColumn = reshape( ...
    string(algorithmNames(A(:))),[],1);

receiverColumn = reshape( ...
    string(receiverNames(R(:))),[],1);

trialsColumn = ...
    repmat(numTrials,numel(S),1);

offeredBitsColumn = ...
    repmat(offeredBits,numel(S),1);

berTable = table( ...
    snrColumn, ...
    channelColumn, ...
    algorithmColumn, ...
    receiverColumn, ...
    trialsColumn, ...
    offeredBitsColumn, ...
    totalBitErrors(:), ...
    decodedBits(:), ...
    lostBits(:), ...
    conditionalBER(:), ...
    frameFailProbability(:), ...
    bitDeliveryFailureRatio(:), ...
    safeFrameCount(:), ...
    unsafeFrameCount(:), ...
    safeBER(:), ...
    unsafeBER(:), ...
    'VariableNames',{ ...
    'SNR_dB', ...
    'Channel', ...
    'Algorithm', ...
    'Receiver', ...
    'Trials', ...
    'Offered_Bits', ...
    'Bit_Errors', ...
    'Decoded_Bits', ...
    'Lost_Bits', ...
    'Conditional_BER', ...
    'Frame_Fail', ...
    'Bit_Delivery_Failure', ...
    'CPsafe_Frame_Count', ...
    'CPunsafe_Frame_Count', ...
    'CPsafe_BER', ...
    'CPunsafe_BER'});

berTable = sortrows(berTable, ...
    {'Channel','SNR_dB','Algorithm','Receiver'});

%% 13. Construct timing summary table

[St,Ct,At] = ndgrid( ...
    1:numSNR, ...
    1:numChannels, ...
    1:numAlgorithms);

timingTable = table( ...
    reshape(snrList(St(:)),[],1), ...
    reshape(string(channelNames(Ct(:))),[],1), ...
    reshape(string(algorithmNames(At(:))),[],1), ...
    repmat(numTrials,numel(St),1), ...
    exactProbability(:), ...
    cpSafeProbability(:), ...
    timingFailProbability(:), ...
    conditionalMSE(:), ...
    meanTimingError(:), ...
    'VariableNames',{ ...
    'SNR_dB', ...
    'Channel', ...
    'Algorithm', ...
    'Trials', ...
    'Exact', ...
    'CPsafe', ...
    'Timing_Fail', ...
    'Conditional_MSE', ...
    'Mean_Timing_Error'});

timingTable = sortrows(timingTable, ...
    {'Channel','SNR_dB','Algorithm'});

assert(height(berTable)== ...
    numSNR*numChannels*numAlgorithms*numReceivers);

assert(height(timingTable)== ...
    numSNR*numChannels*numAlgorithms);

%% 14. Create output directories

dataDir = fullfile( ...
    projectRoot,'results','data');

tableDir = fullfile( ...
    projectRoot,'results','tables');

figureDir = fullfile( ...
    projectRoot,'results','figures');

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

if ~exist(tableDir,'dir')
    mkdir(tableDir);
end

if ~exist(figureDir,'dir')
    mkdir(figureDir);
end

%% 15. Experiment methodology

experimentVersion = ...
    'EXP08 Formal BER Comparison v1.0';

snrDefinition = ...
    'Received active-frame sample SNR';

frameDefinition = ...
    'One 80-sample training symbol and one 80-sample data symbol';

receiverDefinition = { ...
    'True timing and known physical channel', ...
    'Estimated timing and known physical channel', ...
    'Estimated timing and oracle timing-phase compensation'};

randomizationPolicy = ...
    ['Shared arrival delay, data bits and base noise; ' ...
    'algorithm-specific training symbols'];

zeroBERPolicy = ...
    ['Raw zero BER retained; zero observations shown ' ...
    'at one divided by decoded bits for visualization only'];

oracleWarning = ...
    ['True timing error is available only to the ' ...
    'oracle receiver and evaluation code'];

%% 16. Save formal results BEFORE plotting

matPath = fullfile(dataDir, ...
    'exp08_ber_monte_carlo.mat');

save(matPath, ...
    'cfg','experimentVersion', ...
    'snrDefinition','frameDefinition', ...
    'receiverDefinition','randomizationPolicy', ...
    'zeroBERPolicy','oracleWarning', ...
    'randomSeed','arrivalRange','searchRange', ...
    'snrList','numTrials','algorithmNames', ...
    'channelNames','receiverNames', ...
    'channelResponses','Hlist', ...
    'cpSafeMin','bufferLength', ...
    'bitsPerOFDM','offeredBits', ...
    'allArrivalDelays','allTimingErrors', ...
    'allTimingValid','allCPsafe', ...
    'allEstimatedStarts','allPeakMetrics', ...
    'allSignalPowers','allBitErrors', ...
    'allReceiverValid','exactProbability', ...
    'cpSafeProbability','timingFailProbability', ...
    'conditionalMSE','meanTimingError', ...
    'conditionalBER','frameFailProbability', ...
    'totalBitErrors','decodedBits','lostBits', ...
    'bitDeliveryFailureRatio', ...
    'safeBER','unsafeBER', ...
    'safeFrameCount','unsafeFrameCount', ...
    'berTable','timingTable','elapsedTime', ...
    '-v7.3');

berCSVPath = fullfile(tableDir, ...
    'exp08_ber_monte_carlo.csv');

timingCSVPath = fullfile(tableDir, ...
    'exp08_timing_statistics.csv');

writetable(berTable,berCSVPath);

writetable(timingTable,timingCSVPath);

fprintf('\nMAT and CSV results saved successfully.\n');

%% 17. Plot settings

displayNames = { ...
    'Schmidl-Cox','Park','Minn','CAZAC'};

channelLabels = { ...
    'AWGN','Fixed Multipath + AWGN'};

channelTags = {'awgn','multipath'};

receiverLabels = { ...
    'Perfect Timing', ...
    'Estimated Timing', ...
    'Oracle Phase Compensation'};

receiverTags = { ...
    'perfect','estimated','oracle'};

colors = [ ...
    0.0000 0.4470 0.7410; ...
    0.8500 0.3250 0.0980; ...
    0.4660 0.6740 0.1880; ...
    0.4940 0.1840 0.5560];

markers = {'o','s','^','d'};

lineStyles = {'-','--','-.',':'};

receiverColors = [ ...
    0.0000 0.4470 0.7410; ...
    0.8500 0.3250 0.0980; ...
    0.4660 0.6740 0.1880];

receiverMarkers = {'o','s','^'};

receiverStyles = {'-','--','-.'};

%% 18. Handle zero BER for visualization ONLY

BERplot = conditionalBER;

zeroMask = ...
    conditionalBER==0 & decodedBits>0;

BERplot(zeroMask) = ...
    1./decodedBits(zeroMask);

plotFloor = 1/offeredBits;

% Approximate coherent QPSK reference for sample SNR.
% Frame-by-frame CP energy normalization causes
% small deviations from this simple expression.

theoryAWGN = 0.5*erfc( ...
    sqrt(10.^(snrList/10)/2));

figureFiles = {};

%% 19. Four-algorithm BER comparison figures
% Two channels x three receiver types = six figures

for c = 1:numChannels

    for r = 1:numReceivers

        fig = figure('Color','w', ...
            'Position',[100 100 1100 650]);

        hold on;

        for a = 1:numAlgorithms

            y = BERplot(:,c,a,r);

            semilogy(snrList,y, ...
                'Color',colors(a,:), ...
                'LineStyle',lineStyles{a}, ...
                'Marker',markers{a}, ...
                'LineWidth',1.8, ...
                'MarkerSize',6, ...
                'DisplayName',displayNames{a});

            % Mark observed zero BER with downward triangles

            z = zeroMask(:,c,a,r);

            if any(z)

                semilogy(snrList(z),y(z), ...
                    'v', ...
                    'Color',colors(a,:), ...
                    'MarkerSize',8, ...
                    'LineStyle','none', ...
                    'HandleVisibility','off');

            end

        end

        if c==1

            semilogy(snrList,theoryAWGN, ...
                'k--', ...
                'LineWidth',1.2, ...
                'DisplayName', ...
                'Approx. coherent QPSK theory');

        end

        grid on;
        box on;

        set(gca,'YScale','log');

        xlabel('Received Active-frame SNR (dB)');

        ylabel('Conditional BER');

        title(sprintf('%s: %s', ...
            receiverLabels{r}, ...
            channelLabels{c}));

        subtitle(['Downward triangles indicate ' ...
            'zero observed bit errors']);

        legend('Location','best');

        xlim([min(snrList),max(snrList)]);

        xticks(snrList);

        ylim([plotFloor/2,1]);

        set(gca,'FontSize',12);

        figureName = sprintf( ...
            'exp08_%s_ber_%s.png', ...
            channelTags{c},receiverTags{r});

        figurePath = ...
            fullfile(figureDir,figureName);

        exportgraphics(fig,figurePath, ...
            'Resolution',300);

        figureFiles{end+1} = figurePath;

        close(fig);

    end

end

%% 20. Per-algorithm receiver comparison
% Compare perfect timing, estimated timing,
% and oracle compensation.
% Two channels x four algorithms = eight figures

for c = 1:numChannels

    for a = 1:numAlgorithms

        fig = figure('Color','w', ...
            'Position',[100 100 1100 650]);

        hold on;

        for r = 1:numReceivers

            y = BERplot(:,c,a,r);

            semilogy(snrList,y, ...
                'Color',receiverColors(r,:), ...
                'LineStyle',receiverStyles{r}, ...
                'Marker',receiverMarkers{r}, ...
                'LineWidth',1.8, ...
                'MarkerSize',6, ...
                'DisplayName',receiverLabels{r});

            z = zeroMask(:,c,a,r);

            if any(z)

                semilogy(snrList(z),y(z), ...
                    'v', ...
                    'Color',receiverColors(r,:), ...
                    'MarkerSize',8, ...
                    'LineStyle','none', ...
                    'HandleVisibility','off');

            end

        end

        grid on;
        box on;

        set(gca,'YScale','log');

        xlabel('Received Active-frame SNR (dB)');

        ylabel('Conditional BER');

        title(sprintf('%s BER: %s', ...
            displayNames{a}, ...
            channelLabels{c}));

        subtitle(['Downward triangles indicate ' ...
            'zero observed bit errors']);

        legend('Location','best');

        xlim([min(snrList),max(snrList)]);

        xticks(snrList);

        ylim([plotFloor/2,1]);

        set(gca,'FontSize',12);

        figureName = sprintf( ...
            'exp08_%s_%s_receiver_comparison.png', ...
            channelTags{c}, ...
            lower(algorithmNames{a}));

        figurePath = ...
            fullfile(figureDir,figureName);

        exportgraphics(fig,figurePath, ...
            'Resolution',300);

        figureFiles{end+1} = figurePath;

        close(fig);

    end

end

%% 21. Display timing results

for c = 1:numChannels

    fprintf('\n%s Timing Results\n', ...
        channelLabels{c});

    fprintf('%5s %-8s %10s %10s %10s %12s\n', ...
        'SNR','Method','Exact','CP-safe', ...
        'Fail','MSE');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf(['%5d %-8s %10.4f %10.4f ' ...
                '%10.4f %12.4f\n'], ...
                snrList(s), ...
                algorithmNames{a}, ...
                exactProbability(s,c,a), ...
                cpSafeProbability(s,c,a), ...
                timingFailProbability(s,c,a), ...
                conditionalMSE(s,c,a));

        end

    end

end

%% 22. Display BER results

for c = 1:numChannels

    fprintf('\n%s BER Results\n', ...
        channelLabels{c});

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

%% 23. Display actual bit error counts

for c = 1:numChannels

    fprintf('\n%s Bit Error Counts\n', ...
        channelLabels{c});

    fprintf('%5s %-8s %12s %12s %12s\n', ...
        'SNR','Method', ...
        'Perfect','Estimated','Oracle');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf('%5d %-8s %12d %12d %12d\n', ...
                snrList(s), ...
                algorithmNames{a}, ...
                totalBitErrors(s,c,a,1), ...
                totalBitErrors(s,c,a,2), ...
                totalBitErrors(s,c,a,3));

        end

    end

end

%% 24. Display CP-safe and unsafe oracle BER

for c = 1:numChannels

    fprintf('\n%s Oracle BER by Timing Region\n', ...
        channelLabels{c});

    fprintf(['%5s %-8s %10s %10s ' ...
        '%12s %12s\n'], ...
        'SNR','Method','SafeN','UnsafeN', ...
        'SafeBER','UnsafeBER');

    for s = 1:numSNR

        for a = 1:numAlgorithms

            fprintf(['%5d %-8s %10d %10d ' ...
                '%12.6e %12.6e\n'], ...
                snrList(s), ...
                algorithmNames{a}, ...
                safeFrameCount(s,c,a,3), ...
                unsafeFrameCount(s,c,a,3), ...
                safeBER(s,c,a,3), ...
                unsafeBER(s,c,a,3));

        end

    end

end

%% 25. Save plotting data

save(matPath, ...
    'BERplot','zeroMask','plotFloor', ...
    'theoryAWGN','figureFiles','-append');

%% 26. Final completion report

assert(isfile(matPath));

assert(isfile(berCSVPath));

assert(isfile(timingCSVPath));

assert(numel(figureFiles)==14);

assert(all(cellfun(@isfile,figureFiles)));

fprintf('\nEXP08 Completion Summary\n');

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

fprintf('Total Decoded Bits = %.0f\n', ...
    sum(decodedBits(:)));

fprintf('Total Lost Bits = %.0f\n', ...
    sum(lostBits(:)));

fprintf('Offered Bits per Group = %d\n', ...
    offeredBits);

fprintf('BER Table Rows = %d\n', ...
    height(berTable));

fprintf('Timing Table Rows = %d\n', ...
    height(timingTable));

fprintf('Saved Figures = %d\n', ...
    numel(figureFiles));

fprintf('Simulation Time = %.2f seconds\n', ...
    elapsedTime);

fprintf('\nSaved MAT:\n%s\n',matPath);

fprintf('\nSaved BER CSV:\n%s\n',berCSVPath);

fprintf('\nSaved Timing CSV:\n%s\n',timingCSVPath);

fprintf('\nSaved Figures:\n');

for j = 1:numel(figureFiles)

    fprintf('%s\n',figureFiles{j});

end

fprintf('\nAll formal BER checks passed.\n');

fprintf('EXP08 formal BER Monte Carlo completed.\n');
