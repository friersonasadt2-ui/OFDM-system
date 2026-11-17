function [P, E, metric, candidateIndex] = minn_metric(rx, Nfft)
% MINN_METRIC
% Minn et al. (2003), Eq. (14)-(16)
% Four-part timing metric, L = 4.
% rx: received complex sample vector
% Nfft: useful training length (64)
% P: weighted correlation
% E: candidate-window energy
% metric: normalized Minn timing metric
% candidateIndex: 1-based candidate window starts

rx = rx(:);

L = 4;
assert(mod(Nfft,L)==0, ...
    'Nfft must be divisible by 4.');
assert(numel(rx)>=Nfft, ...
    'Received buffer is too short.');

M = Nfft/L;

% Training pattern: [-A, +A, -A, -A]
p = [-1, 1, -1, -1];
b = p(1:end-1).*p(2:end);

candidateIndex = (1:numel(rx)-Nfft+1).';

numCandidates = numel(candidateIndex);

P = zeros(numCandidates,1);
E = zeros(numCandidates,1);
metric = zeros(numCandidates,1);

for i = 1:numCandidates

    d = candidateIndex(i);

    % Extract one Nfft-sample window
    w = rx(d:d+Nfft-1);

    % Divide into four equal parts
    parts = reshape(w,M,L);

    % Eq. (15): weighted adjacent-part correlation
    corrValue = 0;

    for k = 1:L-1
        corrValue = corrValue + b(k) * ...
            sum(conj(parts(:,k)).*parts(:,k+1));
    end

    P(i) = corrValue;

    % Eq. (16): total window energy
    E(i) = sum(abs(w).^2);

    % Eq. (14): normalized timing metric
    if E(i)>1e-12
        metric(i) = ...
            (L/(L-1)*abs(P(i))/E(i))^2;
    else
        metric(i) = 0;
    end

end

end