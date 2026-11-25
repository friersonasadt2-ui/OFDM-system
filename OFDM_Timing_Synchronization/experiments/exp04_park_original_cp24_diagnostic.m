%% EXP04 Original Park competing peaks with altered CP (NOISELESS)
% Diagnostic only. CP=24 is NOT the four-method benchmark setting.
clearvars; clc;
projectRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));
cfg=default_config(); rng(cfg.randomSeed);
cfg.Ncp=24;
cfg.trainingLength=cfg.Nfft+cfg.Ncp;
cfg.trainingEnergy=cfg.trainingPower*cfg.trainingLength;
N=cfg.Nfft; Ncp=cfg.Ncp; D=cfg.arrivalDelay;
[training,~,~,~]=park_training_original(cfg);
idxQpsk=randi([0 3],N,1);
dataFreq=exp(1j*(pi/4+pi/2*idxQpsk));
dataTime=ifft(dataFreq,N);
dataSymbol=[dataTime(end-Ncp+1:end);dataTime];
dataSymbol=normalize_training_energy(dataSymbol,cfg.trainingEnergy);
rx=[zeros(D,1);training;dataSymbol;zeros(N,1)];
trueStart=D+Ncp+1;
expectedCenter=trueStart+N/4;
competingCenter=expectedCenter+16;
[P,R,M,centers]=park_metric_paper2003(rx,N);

fprintf('Original Park CP=24 noiseless diagnostic\n');
fprintf('Training length=%d; energy=%.8f\n',numel(training),sum(abs(training).^2));
fprintf('True FFT start=%d, expected center=%d\n',trueStart,expectedCenter);
for d=[expectedCenter,competingCenter]
    i=find(centers==d,1);
    if isempty(i); error('Missing center %d',d); end
    k=(0:N/2).'; EL=sum(abs(rx(d-k)).^2); ER=sum(abs(rx(d+k)).^2);
    fprintf('Center %d: |P|=%.9f EL=%.9f ER=%.9f OriginalMetric=%.9f\n', ...
        d,abs(P(i)),EL,ER,M(i));
end
arrivalRange=[20 80]; energyThreshold=8;
[estimatedStart,peakCenter,peakMetric,isValid]= ...
    park_estimator_paper2003(rx,cfg,arrivalRange,energyThreshold);
fprintf('Valid=%d Center=%g EstimatedStart=%g Error=%+g PeakMetric=%.8f\n', ...
    isValid,peakCenter,estimatedStart,estimatedStart-trueStart,peakMetric);
fig=figure('Color','w');
plot(centers,M,'b-','LineWidth',1.3); hold on;
xline(expectedCenter,'g--','Expected center');
xline(competingCenter,'r:','Competing center');
xlim([expectedCenter-20,expectedCenter+32]);
xlabel('Correlation Center Index d'); ylabel('Original Park Timing Metric');
title('Original Park CP=24 Noiseless Competing-Peak Diagnostic');
grid on;
figureDir=fullfile(projectRoot,'results','figures');
if ~exist(figureDir,'dir'); mkdir(figureDir); end
exportgraphics(fig,fullfile(figureDir,'exp04_original_park_cp24_peaks.png'),'Resolution',200);
