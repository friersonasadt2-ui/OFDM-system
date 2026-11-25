function [training, useful, A, training_freq] = park_training_original(cfg)
% PARK_TRAINING_ORIGINAL
% Supervisor-confirmed four-part Park training structure.
% Useful training structure:
% [A; B; conj(A); conj(B)]
% where B = flipud(A).

N = cfg.Nfft;
Ncp = cfg.Ncp;

assert(mod(N,4)==0, ...
    'Nfft must be divisible by four.');

assert(Ncp>=0 && Ncp<N, ...
    'Invalid CP length.');

% 1. Generate frequency-domain QPSK symbols
training_freq = qammod( ...
    randi([0 3],N/4,1), ...
    4,'UnitAveragePower',true);

% 2. Construct the four-part training sequence
A = ifft(training_freq,N/4);
B = flipud(A);
C = conj(A);
D = conj(B);

useful = [A; B; C; D];

% 3. Insert cyclic prefix
training = [useful(end-Ncp+1:end); useful];

% 4. Normalize total training energy
training = normalize_training_energy( ...
    training,cfg.trainingEnergy);

% 5. Verify length
assert(numel(training)==N+Ncp, ...
    'Training length mismatch.');

end