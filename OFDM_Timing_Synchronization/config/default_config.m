function cfg = default_config()
%default_configuration for OFDM timing synchronization FYP
% cfg = default_config() returns the basic simulation parameters used
% throughout the project

% OFDM and modulation (preserve existing field names)
cfg.Nfft = 64;
cfg.Ncp = 16;
cfg.M = 4;
cfg.bitsPerSymbol = log2(cfg.M);
cfg.numOFDMSymbols = 1000; % Legacy EXP01/EXP02 setting

% Timing and Monte Carlo
cfg.arrivalDelay = 50;
cfg.SC_threshold = 0.90;
cfg.SNR_dB = 0:2:20;
cfg.numTrials = 10000;
cfg.randomSeed = 42;
cfg.exampleSNR_dB = 10;

% Shared preamble constraint: Nfft useful samples + Ncp CP samples
cfg.trainingLength = cfg.Nfft + cfg.Ncp;
cfg.trainingPower = 1; % Average power per preamble sample
cfg.trainingEnergy = cfg.trainingPower * cfg.trainingLength;

% Channel models; fixed multipath has unit tap energy
cfg.hAWGN = 1;
h = [1; 0.5; 0.25];
cfg.hMultipath = h / norm(h);
cfg.channelNames = {'AWGN', 'Fixed_Multipath'};
cfg.channelResponses = {cfg.hAWGN, cfg.hMultipath};

% Noise variance = received clean active-frame power / 10^(SNR_dB/10).
% The noise calculation must exclude the leading arrival-delay zeros.
cfg.snrDefinition = 'received_active_frame_power';

assert(mod(cfg.Nfft, 2) == 0 && cfg.Ncp < cfg.Nfft);
assert(abs(norm(cfg.hMultipath)^2 - 1) < 1e-12);
end