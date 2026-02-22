% RUN AFTER first running compile_RT_{site}.m for each site

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILD DATABASE FOR MULTI-LEVEL ANALYSIS FOR HABITS-replication of behavioral findings of Tricomi et al.,
% 2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last modified on JUNE 2018 by Eva
% Edited by Rani on Jun 2025

% Description of the exclusions as taken form the R code in the github of the original paper:
% FREEOPERANT <- subset (FREEOPERANT,!ID == '234') # caltech 2 extream
% FREEOPERANT <- subset (FREEOPERANT,!ID == '299'  & !ID == '334' & !ID == '341' & !ID == '310' & !ID == '304' & !ID == '322' & !ID == '326' & !ID == '352' & !ID == '356' & !ID == '360' & !ID == '301') # automated exclusions in Telaviv


close all
clear all

%% INPUT VARIABLE
cd(fileparts(mfilename('fullpath')))

analysss_names     = {'CALTECH_V1_RUN1_FEATURES', 'HAMBURG_RUN1_FEATURES', 'CALTECH_V2_RUN1_FEATURES', 'SYDNEY_RUN1_FEATURES', 'TELAVIV_RUN1_FEATURES'};

% get homedirectory
current_dir       = pwd;
where_to_cut      = (regexp (pwd, 'ANALYSIS') -1);
homedir           = current_dir(1:where_to_cut(end));


questionnaire_dir = fullfile(homedir,'DATA/behavior_telaviv/phenotype');
behavior_dir      = fullfile(homedir,'DATA/behavior_telaviv');
analysis_dir      = fullfile(homedir,'ANALYSIS/compile_databases');

%outputs
databases_dir     = fullfile(homedir, 'DATABASES');

%tools
addpath (genpath(fullfile(analysis_dir,'/my_tools')));

cd (databases_dir)

all_data_RT_table = table;
all_data_DV_table = table;
all_data_run1_features_table = table;
for i = 1:length(analysss_names)
    analysis_name = analysss_names{i};
    disp(['Processing: ' analysis_name]);
    run1_features_table = readtable([analysis_name '.csv']);
    % read csv table
    disp(['Number of rows in current table: ' num2str(height(run1_features_table))]);
    disp(['Number of unique subjects in current table: ' num2str(length(unique(run1_features_table.newID)))]);
    
    % concatenate
    all_data_run1_features_table = [all_data_run1_features_table; run1_features_table];
    
end

% remove exclusions (according to the original paper)
all_data_run1_features_table = all_data_run1_features_table(~ismember(all_data_run1_features_table.newID, [234, 299, 334, 341, 310, 304, 322, 326, 352, 356, 360, 301]), :);
% Save the concatenated table
writetable(all_data_run1_features_table, fullfile(databases_dir, 'RUN1_FEATURES_ALL_SITES.csv'));