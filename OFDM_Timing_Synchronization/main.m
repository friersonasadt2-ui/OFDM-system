%% OFDM Timing Synchronization FYP
% Main project initialization script
clear;
clc;
close all;

%% 1. Locate Project Root
projectRoot = fileparts(mfilename('fullpath'));

%% 2. Add Project Folders to MATLAB Path
addpath(genpath(projectRoot));

%% 3. Initialize Project Paths
paths = project_paths();

%% 4. Display Project Information
fprintf('OFDM Timing Synchronization FYP\n');

fprintf('Project root:\n%s\n\n', paths.root);

fprintf('Figures directory:\n%s\n\n', paths.figures);

fprintf('Tables directory:\n%s\n\n', paths.tables);

fprintf('Data directory:\n%s\n\n', paths.data);

fprintf('Project initialized successfully.\n');