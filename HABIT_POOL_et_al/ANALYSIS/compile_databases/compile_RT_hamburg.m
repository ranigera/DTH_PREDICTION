%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILD DATABASE FOR MULTI-LEVEL ANALYSIS FOR HABITS-replication of behavioral findings of Tricomi et al.,
% 2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last modified on MAY 2018 by Eva
% Edited by Rani on Jun 2025

close all
clear all

%% INPUT VARIABLE
cd(fileparts(mfilename('fullpath')))

analysis_name     = 'RT_HAMBURG';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% participants

subj              = { '06'; '07'; '08'; '09'; '10'; '11'; '12'; '13'; '14'; '15'; '16'; '17'; '18'; '19'; '21'; '22'; '23'; '25'; '26'; '27'; '28'; '29'; '30'; '31'; '32'; '33'; '34'; '35'; '36'; '37'; '38'; '40'; '41'; '42'; '43'; '46'; '48'; '49'; '50'; '52'; '53'; '54'; '55'; '56'; '57'; '58'; '59'; '60'; '61'; '62';  '63';  '64';  '65';  '66';  '67';  '68';  '70';  '72';  '73';  '74';  '75';  '76';  '77';   '78';  '79'}; % participant ID
newID             = {'113';'114';'115';'116';'117';'118';'119';'120';'121';'122';'123';'124';'125';'126';'127';'128';'129';'130';'131';'132';'133';'134';'135';'136';'137';'138';'139';'140';'141';'142';'143';'144';'145';'146';'147';'148';'149';'150';'151';'152';'153';'154';'155';'156';'157';'158';'159';'160';'161';'162'; '163'; '164'; '165'; '166'; '167'; '168'; '169'; '170'; '171'; '172'; '173'; '174'; '175';  '176'; '177'};
group             = [  1     1     1     1     2     1     1     1     1     1     1     2     1     1     1     1     1     1     2     2     2     2     2     2     1     2     2     2     2     1     1     1     2     2     1     1     2     1     1     1     1     2     2     1     1     2     2     2     2     2      2      2      2      2      2      2      2      2      2      2      2      2      2       2       2 ] ; % group

site              = repmat({'Hamburg'},length (subj),1);

% get homedirectory
current_dir       = pwd;
where_to_cut      = (regexp (pwd, 'ANALYSIS') -1);
homedir           = current_dir(1:where_to_cut(end));


questionnaire_dir = fullfile(homedir,'DATA/behavior_hamburg/phenotype');
behavior_dir      = fullfile(homedir,'DATA/behavior_hamburg');
analysis_dir      = fullfile(homedir,'ANALYSIS/compile_databases');

%outputs
databases_dir     = fullfile(homedir, 'DATABASES');

%tools
addpath (genpath(fullfile(analysis_dir,'/my_tools')));


%% LOOP TO EXTRACT DATA
db.devalued = {};

main_DV_table = table;
main_RT_table = table;
for  i=1:length(subj)
    
    clear DATA
    
    clear consumption
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load data
    
    subjX=char(subj(i,1)); % which subject?
    newIDX=char(newID(i,1));
    disp (['******************************* PARTICIPANT: ' subjX ' ***************************************']);
    groupX = group(i); % which group did the subject belong?
    % if not in directory, go to the right folder
    % if ~exist(behavior_dir, 'dir')
    cd (behavior_dir) % go to behav folder to load subject data
    % end
    
    switch groupX % get the specifics according to the group (1 vs 3 days)
        case 1
            session   = {'01'};
            task_name = 'HAB1day';
            groupName = {'1-day'};%{'1'};%
        case 2
            session = {'01'; '02'; '03'};
            task_name = 'HAB3day';
            groupName = {'3-day'};%{'3'};%
    end
    
    siteX= char(site(i)); % get the specific of the site of the data collection
    
    for ii = 1:length(session)
        sessionX = char(session(ii,1));
        load (['sub-' num2str(subjX) '_task-' task_name '_session-' num2str(sessionX(end-1:end)) '.mat'])
        DATA.(['day' num2str(ii)]) = data;
    end
    
    cd (analysis_dir) % go back to analysis folder
    
    trial = 1;
    
    days = fieldnames(DATA);
    for d = days'
        analyzed_day = d{1};
        disp(['Processing ' analyzed_day ' data for subject ' subjX]);
        
        trainig_RTs = {DATA.(analyzed_day).training.blockDetails.RT};
        correct_response = {DATA.(analyzed_day).training.blockDetails.ACC};
        block_cond = DATA.(analyzed_day).training.value;
        block_n = DATA.(analyzed_day).training.block;
        run = DATA.(analyzed_day).training.run;
        session_n = DATA.(analyzed_day).training.session;
        first_RT = DATA.(analyzed_day).training.stPressRT;
        press_freq = DATA.(analyzed_day).training.pressFreq;
        
        if strcmp(groupName{1},'1-day') || (strcmp(groupName{1},'3-day') && DATA.(analyzed_day).training.session(end) == 3)
            % trainnig last run:
            last_run = max(DATA.(analyzed_day).training.run);
            last_run_condition_indicator = DATA.(analyzed_day).training.value(DATA.(analyzed_day).training.run==last_run);
            last_run_press_freq = DATA.(analyzed_day).training.pressFreq(DATA.(analyzed_day).training.run==last_run);
            last_run_deval_press_freq_mean = mean(last_run_press_freq(strcmp(last_run_condition_indicator,'devalued')));
            last_run_still_val_press_freq_mean = mean(last_run_press_freq(strcmp(last_run_condition_indicator,'valued')));
            % extinction:
            extinction_deval_press_freq_mean = mean(DATA.(analyzed_day).extinction.pressFreq(strcmp(DATA.(analyzed_day).extinction.value,'devalued')));
            extinction_still_val_press_freq_mean = mean(DATA.(analyzed_day).extinction.pressFreq(strcmp(DATA.(analyzed_day).extinction.value,'valued')));
            % crate a table:
            DV_table = table;
            DV_table.oldSubID    = str2num(subjX);
            DV_table.newID   = str2num(newIDX);
            DV_table.group   = groupName{1};
            DV_table.site    = siteX;
            % DV_table.day    = session_n(1);
            % DV_table.block = block_n(1);
            % DV_table.run = run(1);
            DV_table.pre_val = last_run_still_val_press_freq_mean;
            DV_table.pre_deval = last_run_deval_press_freq_mean;
            DV_table.post_val = extinction_still_val_press_freq_mean;
            DV_table.post_deval = extinction_deval_press_freq_mean;
            DV_table.CHANGE_SCORE = (DV_table.post_val - DV_table.pre_val) - (DV_table.post_deval - DV_table.pre_deval);
            
            main_DV_table = [main_DV_table; DV_table];
        end
        
        % create a table with the subject number, the session, the site, the group and the devalued snack:
        for ind = 1:length(trainig_RTs)
            RT_table = table;
            if ~isnan(trainig_RTs{ind}) & ~strcmp(block_cond{ind}, 'baseline')
                RTs = trainig_RTs{ind};
                Corr = correct_response{ind};
                % Keep only the trials with RTs and correct responses
                RTs = RTs(Corr==1);
                RT_table.oldSubID    = repmat(str2num(subjX),length(RTs),1);
                RT_table.newID   = repmat(str2num(newIDX),length(RTs),1);
                RT_table.group   = repmat(groupName{1},length(RTs),1);
                RT_table.site    = repmat(siteX,length(RTs),1);
                RT_table.day    = repmat(session_n(ind),length(RTs),1);
                RT_table.block = repmat(block_n(ind),length(RTs),1);
                RT_table.run = repmat(run(ind),length(RTs),1);
                RT_table.first_RT = repmat(first_RT(ind),length(RTs),1);
                RT_table.press_freq = repmat(press_freq(ind),length(RTs),1);
                RT_table.trial  = repmat(trial,length(RTs),1);
                RT_table.outcome = repmat(block_cond{ind},length(RTs),1);
                RT_table.outcome = cellstr(string(RT_table.outcome));
                RT_table.RT = RTs(:);
                % main_RT_table.outcome = cellstr(string(main_RT_table.outcome));
                
                main_RT_table = [main_RT_table; RT_table];
            end
            trial = trial + 1;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% get the kind of snack that has been devalued
    
end

% make outcome a string variable
main_RT_table.outcome = char(main_RT_table.outcome);

%% print database
cd (databases_dir)

% save the main RT table to csv file
writetable(main_RT_table, [analysis_name '.csv']);
% save the main DV table to csv file
writetable(main_DV_table, [analysis_name '_DV.csv']);