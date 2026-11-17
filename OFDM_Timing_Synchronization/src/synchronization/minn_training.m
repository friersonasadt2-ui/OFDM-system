function [training, useful, A] = minn_training(cfg)
% MINN_TRAINING
% Minn et al. (2003) four-part TD training symbol.
% Pattern: [-A, +A, -A, -A]
% A: length-16 bipolar Golay sequence.
% Outputs:
% training: 80x1 training symbol including CP
% useful: 64x1 useful training samples
% A: 16x1 basic Golay sequence

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = 4;
M = N / L;

assert(N == 64 && Ncp == 16, ...
    'This module currently supports Nfft=64, Ncp=16.');

% 1. Generate a length-16 Golay complementary pair
a = 1;
b = 1;

for k = 1:4
    aNew = [a; b];
    bNew = [a; -b];
    a = aNew;
    b = bNew;
end

% Use one member of the Golay pair
A = a(:);

assert(numel(A) == M, ...
    'Golay sequence length is incorrect.');

% 2. Apply Minn's four-part sign pattern
useful = [-A; A; -A; -A];

% 3. Add the cyclic prefix
cp = useful(end-Ncp+1:end);
training = [cp; useful];

% 4. Normalize total transmitted energy
training = normalize_training_energy( ...
    training, cfg.trainingEnergy);

% Keep useful and A consistent after scaling
scale = training(Ncp+1) / useful(1);
useful = useful * scale;
A = A * scale;

end
