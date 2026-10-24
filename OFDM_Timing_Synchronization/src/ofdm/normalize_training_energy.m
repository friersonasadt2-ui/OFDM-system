function y = normalize_training_energy(x, targetEnergy)
%NORMALIZE_TRAINING_ENERGY Set total energy of a complete CP+preamble.
%   y = normalize_training_energy(x, cfg.trainingEnergy)
%   Preserves the input orientation and relative sample structure.

validateattributes(x, {'numeric'}, {'vector','nonempty','finite'}, mfilename, 'x', 1);
validateattributes(targetEnergy, {'numeric'}, ...
    {'scalar','real','positive','finite'}, mfilename, 'targetEnergy', 2);

energy = sum(abs(x).^2);
if energy <= 0
    error('normalize_training_energy:ZeroEnergy', ...
        'Training sequence must have positive energy.');
end

y = x * sqrt(targetEnergy / energy);
assert(abs(sum(abs(y).^2) - targetEnergy) <= ...
    1e-10 * max(1, targetEnergy), 'Energy normalization failed.');
end
