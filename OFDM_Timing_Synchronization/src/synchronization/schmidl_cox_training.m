function [training, useful, A] = schmidl_cox_training(cfg)
% SCHMIDL_COX_TRAINING
% Schmidl-Cox repeated-half OFDM training symbol.
%
% Useful structure: [A; A]
% Training includes cyclic prefix.
% Total training energy = cfg.trainingEnergy.

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;

assert(mod(N,2)==0, ...
    'Nfft must be even.');

assert(Ncp>=0 && Ncp<N, ...
    'Invalid CP length.');

% 1. Generate random QPSK symbols
dataIdx = randi([0 3],L,1);

qpskSymbols = pskmod(dataIdx,4,pi/4);

% 2. Populate even mathematical subcarriers
X = zeros(N,1);

X(1:2:end) = qpskSymbols;

% 3. IFFT produces two identical halves
useful = ifft(X,N);

% 4. Insert cyclic prefix
cp = useful(end-Ncp+1:end);

training = [cp;useful];

% 5. Normalize total training energy
training = normalize_training_energy( ...
    training,cfg.trainingEnergy);

% 6. Return consistently normalized outputs
useful = training(Ncp+1:end);

A = useful(1:L);

% 7. Basic verification
assert(numel(training)==N+Ncp);

assert(max(abs(useful(1:L) ...
    -useful(L+1:end)))<1e-10);

end
