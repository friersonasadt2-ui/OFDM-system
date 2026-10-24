function paths = project_paths()
% PROJECT_PATHS
% Centralized path management for OFDM Timing Synchronization FYP.
% All paths are absolute and independent of MATLAB's
% current working directory.

%% 1. Locate Project Root
configDir = fileparts(mfilename('fullpath'));

projectRoot = fileparts(configDir);

%% 2. Define Main Directories
paths.root = projectRoot;

paths.config = fullfile(projectRoot, 'config');

paths.src = fullfile(projectRoot, 'src');

paths.experiments = fullfile(projectRoot, 'experiments');

paths.results = fullfile(projectRoot, 'results');

%% 3. Define Result Directories
paths.figures = fullfile(paths.results, 'figures');

paths.tables = fullfile(paths.results, 'tables');

paths.data = fullfile(paths.results, 'data');

%% 4. Create Result Directories
resultDirs = { ...
    paths.results, ...
    paths.figures, ...
    paths.tables, ...
    paths.data ...
    };

for k = 1:numel(resultDirs)

    if ~exist(resultDirs{k}, 'dir')

        mkdir(resultDirs{k});

    end

end

end