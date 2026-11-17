%% EXP04 - Park Training Structure Comparison
% Original frequency construction vs ideal Park-like construction
% Both evaluated using Park (2003) timing metric.

clearvars;
clc;

%% 1. Project setup
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = default_config();
rng(cfg.randomSeed);

N = cfg.Nfft;
Ncp = cfg.Ncp;
L = N/2;
D = cfg.arrivalDelay;

%% 2. Original frequency-domain construction
pn = 2*randi([0 1],L,1)-1;

X = zeros(N,1);
X(1:2:end) = pn;

xOriginal = ifft(X,N);

trainingOriginal = ...
    [xOriginal(end-Ncp+1:end);xOriginal];

trainingOriginal = normalize_training_energy( ...
    trainingOriginal,cfg.trainingEnergy);

%% 3. Ideal Park-like training construction
trainingIdeal = park_like_training(cfg);

%% 4. Generate a common QPSK data symbol
dataIdx = randi([0 3],N,1);
dataFreq = pskmod(dataIdx,4,pi/4);
dataTime = ifft(dataFreq,N);

dataSymbol = ...
    [dataTime(end-Ncp+1:end);dataTime];

dataSymbol = normalize_training_energy( ...
    dataSymbol,cfg.trainingEnergy);

%% 5. Construct two received signals
rxOriginal = [zeros(D,1); ...
    trainingOriginal;dataSymbol;zeros(N,1)];

rxIdeal = [zeros(D,1); ...
    trainingIdeal;dataSymbol;zeros(N,1)];

trueStart = D+Ncp+1;

referenceCenters = trueStart+[L-1,L];

%% 6. Apply SAME Park (2003) timing metric
[~,~,Moriginal,centers] = ...
    park_metric_paper2003(rxOriginal,N);

[~,~,Mideal,centersIdeal] = ...
    park_metric_paper2003(rxIdeal,N);

assert(isequal(centers,centersIdeal));

%% 7. Find and display the five highest peaks
[originalSorted,idxOriginal] = ...
    sort(Moriginal,'descend');

[idealSorted,idxIdeal] = ...
    sort(Mideal,'descend');

fprintf('\nPark Training Structure Comparison\n');
fprintf('True Useful Start = %d\n',trueStart);

fprintf('\nOriginal Construction - Top 5\n');

for k = 1:5
    fprintf('Center %d: %.8f\n', ...
        centers(idxOriginal(k)),originalSorted(k));
end

fprintf('\nIdeal Construction - Top 5\n');

for k = 1:5
    fprintf('Center %d: %.8f\n', ...
        centers(idxIdeal(k)),idealSorted(k));
end

%% 8. Display reference center metrics
fprintf('\nReference Center Metrics\n');
fprintf('Center\tOriginal\tIdeal\n');

for d = referenceCenters

    idx = find(centers==d,1);

    fprintf('%d\t%.8f\t%.8f\n', ...
        d,Moriginal(idx),Mideal(idx));

end

%% 9. Plot comparison
fig = figure('Color','w');

plot(centers,Moriginal,'b-', ...
    'LineWidth',1.3);

hold on;

plot(centers,Mideal,'r-', ...
    'LineWidth',1.3);

xline(referenceCenters(1),'k:', ...
    'LineWidth',1);

xline(referenceCenters(2),'k--', ...
    'LineWidth',1);

xlabel('Candidate Center Index');
ylabel('Timing Metric');

title('Park 2003 Metric: Training Structure Comparison');

legend( ...
    'Original Frequency Construction', ...
    'Ideal Four-Part Construction', ...
    'Center 98', ...
    'Center 99', ...
    'Location','best');

grid on;
xlim([40 220]);

%% 10. Save diagnostic results
figDir = fullfile(projectRoot,'results','figures');
dataDir = fullfile(projectRoot,'results','data');

if ~exist(figDir,'dir')
    mkdir(figDir);
end

if ~exist(dataDir,'dir')
    mkdir(dataDir);
end

exportgraphics(fig,fullfile(figDir, ...
    'exp04_park_structure_comparison.png'), ...
    'Resolution',300);

save(fullfile(dataDir, ...
    'exp04_park_structure_comparison.mat'), ...
    'cfg','centers','Moriginal','Mideal', ...
    'trainingOriginal','trainingIdeal', ...
    'trueStart','referenceCenters');

fprintf('\nComparison results saved.\n');