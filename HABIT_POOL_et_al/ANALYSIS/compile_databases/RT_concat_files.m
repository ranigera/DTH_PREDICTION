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

analysss_names     = {'RT_CALTECH_V1', 'RT_HAMBURG', 'RT_CALTECH_V2', 'RT_SYDNEY', 'RT_TELAVIV'};

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
for i = 1:length(analysss_names)
    analysis_name = analysss_names{i};
    disp(['Processing: ' analysis_name]);
    % read csv table
    RT_table = readtable([analysis_name '.csv']);
    disp(['Number of rows in current table: ' num2str(height(RT_table))]);
    disp(['Number of unique subjects in current table: ' num2str(length(unique(RT_table.newID)))]);
    DV_table = readtable([analysis_name '_DV.csv']);
    
    % concatenate
    all_data_RT_table = [all_data_RT_table; RT_table];
    all_data_DV_table = [all_data_DV_table; DV_table];
end

% remove exclusions (according to the original paper)
all_data_RT_table = all_data_RT_table(~ismember(all_data_RT_table.newID, [234, 299, 334, 341, 310, 304, 322, 326, 352, 356, 360, 301]), :);
all_data_DV_table = all_data_DV_table(~ismember(all_data_DV_table.newID, [234, 299, 334, 341, 310, 304, 322, 326, 352, 356, 360, 301]), :);
% Save the concatenated table
writetable(all_data_RT_table, fullfile(databases_dir, 'RT_ALL_SITES.csv'));
writetable(all_data_DV_table, fullfile(databases_dir, 'DV_ALL_SITES.csv'));