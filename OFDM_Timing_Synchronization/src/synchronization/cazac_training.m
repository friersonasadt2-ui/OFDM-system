function [training, useful, zc] = cazac_training(cfg)
% CAZAC_TRAINING
% Length-64 Zadoff-Chu time-domain training symbol.
% Even-length ZC sequence, root u = 1.
% training: 80x1 symbol including cyclic prefix
% useful: 64x1 useful training samples
% zc: original 64x1 Zadoff-Chu sequence
% This is a representative ZC-based design,
% not a verified reproduction of Fang et al. (2012).

N = cfg.Nfft;
Ncp = cfg.Ncp;
u = 1;

assert(N == 64 && Ncp == 16, ...
    'This implementation uses Nfft=64, Ncp=16.');

assert(gcd(u,N) == 1, ...
    'ZC root must be coprime with sequence length.');

% 1. Generate even-length Zadoff-Chu sequence
n = (0:N-1).';

zc = exp(-1j*pi*u*(n.^2)/N);

% 2. Useful training samples
useful = zc;

% 3. Add cyclic prefix
cp = useful(end-Ncp+1:end);
training = [cp; useful];

% 4. Normalize transmitted energy
energy = sum(abs(training).^2);

assert(energy > 0, 'Training energy must be positive.');

scale = sqrt(cfg.trainingEnergy/energy);

training = training*scale;
useful = useful*scale;

end
