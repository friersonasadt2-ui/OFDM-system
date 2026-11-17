
function trainingSymbol = park_like_training(cfg)
% Generate the validated ideal Park-like four-part preamble.
% This is NOT yet a verified reproduction of Park (2003).

N = cfg.Nfft;

assert(mod(N,4)==0, ...
    'Nfft must be divisible by four.');

Q = N/4;

% Generate QPSK symbols
pn = randi([0 3],Q,1);
A = ifft(pskmod(pn,4,pi/4),Q);

% Construct ideal four-part structure
B = flipud(A);
C = conj(A);
D = conj(B);

x = [A;B;C;D];

% Add cyclic prefix
cp = x(end-cfg.Ncp+1:end);
trainingSymbol = [cp;x];

% Normalize total energy
trainingSymbol = normalize_training_energy( ...
    trainingSymbol,cfg.trainingEnergy);
end
