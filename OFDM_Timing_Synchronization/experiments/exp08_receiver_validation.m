%% EXP08 OFDM Receiver Validation
% Verify modulation, FFT indexing, channel equalization
% and ideal timing-phase compensation.

clearvars; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();

rng(2046);

N = cfg.Nfft;
Ncp = cfg.Ncp;
symbolLength = N+Ncp;
bitsPerOFDM = N*cfg.bitsPerSymbol;

k = (0:N-1).';

fprintf('EXP08 Receiver Validation\n\n');

%% Test 1: QPSK modulation and demodulation

txBits = randi([0 1],bitsPerOFDM,1);

X = qpsk_mod(txBits);
X = X(:);

bitsHat = qpsk_demod(X);
bitsHat = bitsHat(:);

assert(numel(X)==N);
assert(numel(bitsHat)==bitsPerOFDM);

bitErrors = sum(bitsHat~=txBits);

fprintf('Test 1: QPSK Round-trip\n');
fprintf('Transmitted Bits = %d\n',numel(txBits));
fprintf('Bit Errors = %d\n',bitErrors);

assert(bitErrors==0);

fprintf('Test 1 PASSED\n\n');

%% Test 2: OFDM modulation

dataSymbol = ofdm_modulate(X,cfg);
dataSymbol = dataSymbol(:);

assert(numel(dataSymbol)==symbolLength);

Xcheck = ofdm_demodulate(dataSymbol,cfg);
Xcheck = Xcheck(:);

bitsHat = qpsk_demod(Xcheck);
bitsHat = bitsHat(:);

bitErrors = sum(bitsHat~=txBits);

fprintf('Test 2: OFDM Round-trip\n');

fprintf('OFDM Symbol Length = %d\n', ...
    numel(dataSymbol));

fprintf('Bit Errors = %d\n',bitErrors);

assert(bitErrors==0);

% Match the transmission energy used in EXP07
dataSymbol = normalize_training_energy( ...
    dataSymbol,cfg.trainingEnergy);

assert(abs(sum(abs(dataSymbol).^2) ...
    -cfg.trainingEnergy)<1e-10);

fprintf('Normalized Data Energy = %.8f\n', ...
    sum(abs(dataSymbol).^2));

fprintf('Test 2 PASSED\n\n');

%% Construct a transmission frame

D = cfg.arrivalDelay;

training = park_training_original(cfg);

tx = [zeros(D,1); ...
      training; ...
      dataSymbol; ...
      zeros(N,1)];

trueTrainingStart = D+Ncp+1;

trueDataStart = trueTrainingStart+symbolLength;

fprintf('True Training FFT Start = %d\n', ...
    trueTrainingStart);

fprintf('True Data FFT Start = %d\n\n', ...
    trueDataStart);

%% Test 3: Perfect timing, noiseless single path

Hawgn = ones(N,1);

[bitsHat,valid] = ofdm_receive_at_start( ...
    tx,trueDataStart,cfg,Hawgn);

assert(valid);

bitErrors = sum(bitsHat~=txBits);

fprintf('Test 3: Perfect Timing, Noiseless\n');
fprintf('Bit Errors = %d\n',bitErrors);

assert(bitErrors==0);

fprintf('Test 3 PASSED\n\n');

%% Test 4: Perfect timing, fixed multipath

h = cfg.hMultipath(:);

rxMultipath = conv(tx,h);

Hmp = fft(h,N);

assert(all(abs(Hmp)>1e-12));

[bitsHat,valid] = ofdm_receive_at_start( ...
    rxMultipath,trueDataStart,cfg,Hmp);

assert(valid);

bitErrors = sum(bitsHat~=txBits);

fprintf('Test 4: Perfect Timing, Multipath\n');

fprintf('Minimum Channel Magnitude = %.8f\n', ...
    min(abs(Hmp)));

fprintf('Bit Errors = %d\n',bitErrors);

assert(bitErrors==0);

fprintf('Test 4 PASSED\n\n');

%% Test 5: Early timing and ideal compensation

timingError = -7;

estimatedDataStart = trueDataStart+timingError;

channelNames = {'Noiseless','Fixed Multipath'};

rxList = {tx,rxMultipath};

channelList = {Hawgn,Hmp};

fprintf('Test 5: CP-safe Timing Error\n');

fprintf('Timing Error = %+d samples\n', ...
    timingError);

for c = 1:2

    rx = rxList{c};
    H = channelList{c};

    % Case A: Known physical channel only

    [bitsNoComp,valid] = ...
        ofdm_receive_at_start( ...
        rx,estimatedDataStart,cfg,H);

    assert(valid);

    errorsNoComp = sum(bitsNoComp~=txBits);

    % Case B: Ideal effective channel
    % True error is used for validation ONLY.

    phaseRotation = ...
        exp(1j*2*pi*k*timingError/N);

    Heffective = H.*phaseRotation;

    [bitsComp,valid] = ...
        ofdm_receive_at_start( ...
        rx,estimatedDataStart,cfg,Heffective);

    assert(valid);

    errorsComp = sum(bitsComp~=txBits);

    fprintf('\nChannel: %s\n',channelNames{c});

    fprintf('Errors without compensation = %d\n', ...
        errorsNoComp);

    fprintf('BER without compensation = %.8f\n', ...
        errorsNoComp/bitsPerOFDM);

    fprintf('Errors with ideal compensation = %d\n', ...
        errorsComp);

    assert(errorsComp==0, ...
        'Ideal phase compensation failed.');

end

fprintf('\nTest 5 PASSED\n\n');

%% Test 6: Invalid FFT window protection

invalidStarts = [NaN,0,-1,1.5,numel(tx)];

for j = 1:numel(invalidStarts)

    [bitsHat,valid] = ofdm_receive_at_start( ...
        tx,invalidStarts(j),cfg,Hawgn);

    assert(~valid);

    assert(isempty(bitsHat));

end

fprintf('Test 6: Invalid Window Protection\n');

fprintf('Invalid Cases Rejected = %d/%d\n', ...
    numel(invalidStarts),numel(invalidStarts));

fprintf('Test 6 PASSED\n\n');

fprintf('All EXP08 receiver checks passed.\n');
