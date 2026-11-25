%% EXP04 Original Park IFFT symmetry verification (NOISELESS)
% Tests raw IFFT signal: repeated half, circular conjugate relation,
% direct four-part conjugacy, and mirror symmetry at expected center.
clearvars; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));
cfg = default_config(); rng(cfg.randomSeed);
N=cfg.Nfft; Ncp=cfg.Ncp; Q=N/4; L=N/2; D=cfg.arrivalDelay;
[training,~,pn,X] = park_training_original(cfg);
x = ifft(X,N); % unnormalized raw IFFT, to match earlier checks
A=x(1:Q); B=x(Q+1:2*Q); C=x(2*Q+1:3*Q); Dpart=x(3*Q+1:N);
errRepeat=max(abs(x(1:L)-x(L+1:N)));
n=(0:N-1).'; mirrorIdx=mod(-n,N)+1;
errCircular=max(abs(x-conj(x(mirrorIdx))));
errCA=max(abs(C-conj(A)));
errDB=max(abs(Dpart-conj(B)));
errReverse=max(abs(B-flipud(A)));
errConjReverse=max(abs(B-conj(flipud(A))));
rawTraining=[x(end-Ncp+1:end);x];
rx=[zeros(D,1);rawTraining;zeros(N,1)];
trueStart=D+Ncp+1; expectedCenter=trueStart+N/4;
k=(0:L).';
errCenter=max(abs(rx(expectedCenter-k)-conj(rx(expectedCenter+k))));
fprintf('Original Park training symmetry checks (raw IFFT, seed=%d)\n',cfg.randomSeed);
fprintf('Repeated-half error:             %.12e\n',errRepeat);
fprintf('Circular conjugate error:        %.12e\n',errCircular);
fprintf('C - conj(A) error:              %.12e\n',errCA);
fprintf('D - conj(B) error:              %.12e\n',errDB);
fprintf('B - reverse(A) error:           %.12e\n',errReverse);
fprintf('B - conj(reverse(A)) error:     %.12e\n',errConjReverse);
fprintf('Center mirror error at d=%d:    %.12e\n',expectedCenter,errCenter);
fprintf('NOTE: direct four-part equalities are separate from circular symmetry.\n');
