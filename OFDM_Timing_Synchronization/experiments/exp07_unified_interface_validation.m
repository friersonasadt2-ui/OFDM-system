%% EXP07 Unified Interface Validation
% Verify common search coordinates for all four algorithms.
% No AWGN and no multipath.

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

rng(2042);

N = cfg.Nfft;
Ncp = cfg.Ncp;

%% 1. Common experimental assumptions

arrivalRange = [20 80];

margin = Ncp;

trueStartRange = arrivalRange+Ncp+1;

searchRange = [ ...
    trueStartRange(1)-margin, ...
    trueStartRange(2)+margin];

delayList = [20 30 50 70 80];

algorithmNames = {'SC','Park','Minn','CAZAC'};

numAlgorithms = numel(algorithmNames);

numDelays = numel(delayList);

% Fixed receiver buffer length for every arrival delay
bufferLength = arrivalRange(2) ...
    +2*(N+Ncp)+N ...
    +numel(cfg.hMultipath)-1;

%% 2. Generate training symbols

trainingList = { ...
    schmidl_cox_training(cfg), ...
    park_training_original(cfg), ...
    minn_training(cfg), ...
    cazac_training(cfg)};

%% 3. Generate common QPSK data symbol

dataIdx = randi([0 3],N,1);

dataFreq = qammod(dataIdx,4, ...
    'UnitAveragePower',true);

xData = ifft(dataFreq,N);

dataSymbol = [xData(end-Ncp+1:end);xData];

dataSymbol = normalize_training_energy( ...
    dataSymbol,cfg.trainingEnergy);

%% 4. Initialize verification results

allErrors = NaN(numDelays,numAlgorithms);

allValid = false(numDelays,numAlgorithms);

fprintf('EXP07 Unified Interface Validation\n');

fprintf('Arrival Range = [%d, %d]\n',arrivalRange);

fprintf('True Start Range = [%d, %d]\n',trueStartRange);

fprintf('Common Search Range = [%d, %d]\n',searchRange);

fprintf('Receiver Buffer Length = %d\n\n',bufferLength);

fprintf('%-8s %5s %10s %10s %8s %7s %7s\n', ...
    'Method','Delay','TrueStart','Estimate', ...
    'Error','Valid','CPsafe');

%% 5. Test all algorithms and arrival delays

for a = 1:numAlgorithms

    algorithm = algorithmNames{a};

    training = trainingList{a};

    txFrame = [training;dataSymbol];

    assert(abs(sum(abs(training).^2) ...
        -cfg.trainingEnergy)<1e-10);

    for j = 1:numDelays

        D = delayList(j);

        rx = zeros(bufferLength,1);

        rx(D+1:D+numel(txFrame)) = txFrame;

        trueStart = D+Ncp+1;

        [est,~,valid,info] = ...
            timing_estimator_unified( ...
            rx,cfg,algorithm,searchRange);

        allValid(j,a) = valid;

        e = NaN;

        if valid

            e = est-trueStart;

            allErrors(j,a) = e;

            assert(est>=searchRange(1));
            assert(est<=searchRange(2));

        end

        cpSafe = valid && e>=-Ncp && e<=0;

        fprintf('%-8s %5d %10d %10.0f %8.0f %7d %7d\n', ...
            algorithm,D,trueStart,est,e,valid,cpSafe);

        % Check common candidate support

        candidateStarts = ...
            info.usefulIndex(info.searchValid);

        assert(isequal(candidateStarts, ...
            (searchRange(1):searchRange(2)).'));

    end

end

%% 6. Summarize results

fprintf('\nValidation Summary\n');

for a = 1:numAlgorithms

    errors = allErrors(:,a);

    validCount = sum(allValid(:,a));

    exactCount = sum(errors==0);

    cpSafeCount = sum(errors>=-Ncp & errors<=0);

    fprintf('%-8s | Valid = %d/%d | Exact = %d/%d | CP-safe = %d/%d\n', ...
        algorithmNames{a}, ...
        validCount,numDelays, ...
        exactCount,numDelays, ...
        cpSafeCount,numDelays);

end

%% 7. Structural acceptance checks

assert(all(allValid(:)), ...
    'Some noiseless estimates are invalid.');

fprintf('\nAll algorithms returned valid estimates.\n');

fprintf('Common candidate search support verified.\n');

if all(allErrors(:,2:4)==0,'all')

    fprintf('Park, Minn and CAZAC exact timing checks passed.\n');

else

    fprintf('Some peak-based noiseless estimates require review.\n');

end

if all(allErrors(:,1)>=-Ncp ...
        & allErrors(:,1)<=0)

    fprintf('S&C CP-safe timing checks passed.\n');

else

    fprintf('S&C plateau behavior requires review.\n');

end

fprintf('\nEXP07 unified interface validation completed.\n');