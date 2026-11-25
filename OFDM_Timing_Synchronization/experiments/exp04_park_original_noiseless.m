%% EXP04 Original Park PN/IFFT and paper-metric demonstration (NOISELESS)
% Standalone; no Park-like, Early Peak or alternative normalization.
close all;
clearvars; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));
cfg = default_config(); rng(cfg.randomSeed);
N = cfg.Nfft; Ncp = cfg.Ncp; 
D = cfg.arrivalDelay;

% Transmitter: one Park training OFDM symbol, followed by one QPSK symbol.
% [training, useful, pn, X] = park_training_original(cfg);

dataIdx = randi([0 3],N,1);
dataFreq = qammod(dataIdx,4,'UnitAveragePower', true); % QPSK; no toolbox needed
xData = ifft(dataFreq,N);

training_freq  = qammod(randi([0 3],N/4,1),4,'UnitAveragePower', true);
traing_time_A = ifft(training_freq,N/4);
traing_time_B = traing_time_A(end:-1:1,1);
traing_time_C = conj(traing_time_A);
traing_time_D = conj(traing_time_B);
training_time  = [traing_time_A;traing_time_B;traing_time_C;traing_time_D];
training = [training_time(end-Ncp+1:end);training_time];
training = normalize_training_energy(training,cfg.trainingEnergy);

dataSymbol = [xData(end-Ncp+1:end);xData];
dataSymbol = normalize_training_energy(dataSymbol,cfg.trainingEnergy);


rx = [zeros(D,1);training;dataSymbol;zeros(N,1)];
trueUsefulStart = D+Ncp+1;
expectedCenter = trueUsefulStart+N/2;

% Original Park (2003) correlation formula; scan all valid centers.
[P,R,M,centers] = park_metric_paper2003(rx,N);
idx = find(centers==expectedCenter,1);
assert(~isempty(idx),'Expected center is outside valid metric support.');
fprintf('Original Park noiseless diagnostic\n');
fprintf('FFT=%d  CP=%d  Arrival D=%d\n',N,Ncp,D);
fprintf('Training length=%d energy=%.8f\n',numel(training),sum(abs(training).^2));
fprintf('True useful FFT start=%d; expected structural center=%d\n',trueUsefulStart,expectedCenter);
fprintf('Original metric at expected center=%.10f\n',M(idx));
[metricSorted,sortIdx] = sort(M,'descend');
fprintf('Raw global top 5 centers (includes data/trailing buffer):\n');
for k=1:min(5,numel(sortIdx))
    fprintf('  d=%d, M=%.8f\n',centers(sortIdx(k)),metricSorted(k));
end

% Separately report the experimental max-peak estimator; it uses a
% prespecified arrival range and energy gate NOT specified in the paper.
arrivalRange = [20 80]; energyThreshold = 8;
[estimatedStart,peakCenter,peakMetric,isValid] = ...
    park_estimator_paper2003(rx,cfg,arrivalRange,energyThreshold);
fprintf('Experimental max-peak: valid=%d center=%.0f estimate=%.0f peak=%.8f\n', ...
    isValid,peakCenter,estimatedStart,peakMetric);
if isValid
    fprintf('Timing error=%+g samples\n',estimatedStart-trueUsefulStart);
end

fig = figure('Color','w');
plot(centers,M,'b-','LineWidth',1.2); hold on;
xline(expectedCenter,'g--','Expected structural center');
if isValid
    xline(peakCenter,'r:','Selected center');
end
xlabel('Correlation Center Index d'); ylabel('Original Park Timing Metric');
title(sprintf('Original Park PN/IFFT Metric (N=%d, CP=%d, Noiseless)',N,Ncp));
grid on;
figureDir = fullfile(projectRoot,'results','figures');
if ~exist(figureDir,'dir'); mkdir(figureDir); end
exportgraphics(fig,fullfile(figureDir,'exp04_original_park_noiseless.png'),'Resolution',200);
