function rxSignal = add_awgn(txSignal, SNR_dB)
%ADD_AWGN Add complex AWGN to a transmitted signal
% Input:
% txSignal - Transmitted complex baseband signal
% SNR_dB - Signal-to-noise ratio in dB
% Output:
% rxSignal - Received signal with AWGN

%% Calculate signal power
signalPower = mean(abs(txSignal).^2);

%% Convert SNR from dB to linear scale
SNR_linear = 10^(SNR_dB/10);

%% Calculate noise power
noisePower = signalPower / SNR_linear;

%% Generate complex Gaussian noise
noise = sqrt(noisePower/2) * ...
    (randn(size(txSignal)) + 1j*randn(size(txSignal)));

%% Add noise
rxSignal = txSignal + noise;
end