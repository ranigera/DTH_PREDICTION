%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILD DATABASE FOR MULTI-LEVEL ANALYSIS FOR HABITS-replication of behavioral findings of Tricomi et al.,
% 2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last modified on JUNE 2018 by Eva
% Edited by Rani on Jun 2025

close all
clear all

%% INPUT VARIABLE
cd(fileparts(mfilename('fullpath')))

analysis_name     = 'TELAVIV_RUN1_FEATURES';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% participants
subj              = { '101'; '102'; '103'; '104'; '105'; '106'; '107'; '108'; '109'; '110'; '111'; '112'; '113'; '114'; '115'; '116'; '117'; '118'; '119'; '120'; '121'; '122'; '123'; '124'; '125'; '126'; '127'; '128'; '129'; '130'; '131'; '132'; '133'; '134'; '135'; '136'; '201'; '202'; '203'; '204'; '206'; '207'; '208'; '209'; '210'; '211'; '212'; '213'; '214'; '215'; '216'; '217'; '218'; '219'; '220'; '221'; '226'; '227'; '228'; '229'; '230'; '231'; '232'; '233'; '234'; '235'; '237'; '238'; '239'; '240'; '241'}; % participant ID
newID             = { '298'; '299'; '300'; '301'; '302'; '303'; '304'; '305'; '306'; '307'; '308'; '309'; '310'; '311'; '312'; '313'; '314'; '315'; '316'; '317'; '318'; '319'; '320'; '321'; '322'; '323'; '324'; '325'; '326'; '327'; '328'; '329'; '330'; '331'; '332'; '333'; '334'; '335'; '336'; '337'; '338'; '339'; '340'; '341'; '342'; '343'; '344'; '345'; '346'; '347'; '348'; '349'; '350'; '351'; '352'; '353'; '354'; '355'; '356'; '357'; '358'; '359'; '360'; '361'; '362'; '363'; '364'; '365'; '366'; '367'; '368'};
group             = [   1       1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2];

site              = repmat({'Tel_Aviv'},length (subj),1);


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



%% LOOP TO EXTRACT DATA
db.devalued = {};

run1_features_table = table;

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
    % for d =  {'day1'}
    % analyzed_day = d{1};
    analyzed_day = 'day1';
    disp(['Processing ' analyzed_day ' data for subject ' subjX]);
    
    trainig_RTs = {DATA.(analyzed_day).training.blockDetails.RT};
    correct_response = {DATA.(analyzed_day).training.blockDetails.ACC};
    button_pressed = {DATA.(analyzed_day).training.blockDetails.button};
    reward_times = {DATA.(analyzed_day).training.blockDetails.potential_rewards_time};
    block_cond = DATA.(analyzed_day).training.value;
    block_n = DATA.(analyzed_day).training.block;
    run = DATA.(analyzed_day).training.run;
    session_n = DATA.(analyzed_day).training.session;
    first_RT = DATA.(analyzed_day).training.stPressRT;
    press_freq = DATA.(analyzed_day).training.pressFreq;
    value = DATA.(analyzed_day).training.value;
    
    % Now get all of these bu only for the first run:
    trainig_RTs_run1 = trainig_RTs(DATA.(analyzed_day).training.run==1);
    correct_response_run1 = correct_response(DATA.(analyzed_day).training.run==1);
    button_pressed_run1 = button_pressed(DATA.(analyzed_day).training.run==1);
    reward_times_run1 = reward_times(DATA.(analyzed_day).training.run==1);
    block_cond_run1 = block_cond(DATA.(analyzed_day).training.run==1);
    block_n_run1 = block_n(DATA.(analyzed_day).training.run==1);
    session_n_run1 = session_n(DATA.(analyzed_day).training.run==1);
    first_RT_run1 = first_RT(DATA.(analyzed_day).training.run==1);
    press_freq_run1 = press_freq(DATA.(analyzed_day).training.run==1);
    condition_run1 = DATA.(analyzed_day).training.condition(DATA.(analyzed_day).training.run==1);
    value_run1 = value(DATA.(analyzed_day).training.run==1);
    
    % get resp rate mean and std for run 1
    resp_rate_trial_mean = mean(press_freq_run1(~~condition_run1), 'omitnan'); % V
    resp_rate_trial_std = std(press_freq_run1(~~condition_run1), 'omitnan'); % V
    
    resp_rate_cond1 = press_freq_run1(condition_run1==1);
    resp_rate_cond2 = press_freq_run1(condition_run1==2);
    resp_pairs_sum = resp_rate_cond1+resp_rate_cond2;
    resp_pairs_sum_std = std(resp_pairs_sum); % V
    
    % get first RT and std for run 1:
    RTs_first_resp_mean = mean(first_RT_run1(~~condition_run1), 'omitnan'); % V
    RTs_first_resp_std = std(first_RT_run1(~~condition_run1), 'omitnan'); % V
    
    % mean and std of correct responses for run 1
    isCorrectResp_prop_mean = mean(cellfun(@(x) mean(x), correct_response_run1(~~condition_run1)), 'omitnan'); % V
    isCorrectResp_prop_std = std(cellfun(@(x) mean(x), correct_response_run1(~~condition_run1)), 'omitnan'); % V
    
    % compute trend features on resp_pairs_sum:
    resp_pairs_sum_t = table(resp_pairs_sum', 'VariableNames', {'values'});
    time = (1:length(resp_pairs_sum_t.values))';
    % run a linear model
    lm = fitlm(time, resp_pairs_sum_t.values);
    resp_pairs_sum_slope = lm.Coefficients.Estimate(2); % V
    % now Spearman correlation:
    [resp_pairs_sum_spearman_corr, pval] = corr(time, resp_pairs_sum_t.values, 'Type', 'Spearman'); % V
    
    
    % create a table with the subject number, the session, the site, the group and the devalued snack:
    main_RT_table_subject = table;
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
            
            main_RT_table_subject = [main_RT_table_subject; RT_table];
        end
        trial = trial + 1;
    end
    main_RT_table_run1 = main_RT_table_subject(main_RT_table_subject.run==1, :);
    % now add a clumn diff_RT by using diff on RTs per subject
    
    % compute DV only for 1-day group and for 3-day group only if last session is session 3
    main_RT_table_run1.RT_diff = NaN(height(main_RT_table_run1), 1);
    blocks = unique(main_RT_table_run1.block);
    
    for b = 1:length(blocks)
        idx = main_RT_table_run1.block == blocks(b);
        main_RT_table_run1.RT_diff(idx) = [NaN; diff(main_RT_table_run1.RT(idx))];
    end
    
    resp_time_diff_mean = mean(main_RT_table_run1.RT_diff, 'omitnan'); % V
    resp_time_diff_std = std(main_RT_table_run1.RT_diff, 'omitnan'); % V
    
    % now add the abs diff of RT_diff:
    % get the diff of RT_diff per subject:
    volatility = mean(abs(diff(main_RT_table_run1.RT_diff)), 'omitnan'); % V
    
    rel_button_presses = [button_pressed_run1{:}];
    num_f_presses = sum(strcmp(rel_button_presses, 'f'));
    num_d_presses = sum(strcmp(rel_button_presses, 'd'));
    right_vs_left = num_f_presses - num_d_presses; % V
    
    devalued_taste = DATA.(analyzed_day).target;
    % now for valued taste make the opposite of salty/sweet
    if strcmp(devalued_taste, 'salty')
        valued_taste = 'sweet';
    else
        valued_taste = 'salty';
    end
    % now create a variable like value_run1 and replace the 'devalued' with the actual taste
    taste_value_run1 = value_run1;
    taste_value_run1(strcmp(value_run1, 'devalued')) = {devalued_taste};
    taste_value_run1(strcmp(value_run1, 'valued')) = {valued_taste};
    % now for 'valued repleace
    correct_response_run1_sweet = correct_response_run1(strcmp(taste_value_run1, 'sweet'));
    correct_response_run1_salty = correct_response_run1(strcmp(taste_value_run1, 'salty'));
    n_response_run1_sweet = length([correct_response_run1_sweet{:}]);
    n_response_run1_salty = length([correct_response_run1_salty{:}]);
    sweet_vs_salty = n_response_run1_sweet - n_response_run1_salty; % V
    
    non_rest_RTs = trainig_RTs_run1(~~condition_run1);
    non_rest_rewardTimes = reward_times_run1(~~condition_run1);
    post_reward_RTs = [];
    post_reward_RTs_diff = [];
    % iterate non_rest_rewardTimes:
    for r = 1:length(non_rest_rewardTimes)
        reward_times_currentBlock = non_rest_rewardTimes{r};
        RTs_currentBlock = non_rest_RTs{r};
        for rew_time = reward_times_currentBlock
            for rt_ind = 1:length(RTs_currentBlock)
                if RTs_currentBlock(rt_ind) > rew_time
                    post_reward_RTs = [post_reward_RTs; RTs_currentBlock(rt_ind)-rew_time];
                    % check if RTs_currentBlock(rt_ind-1) exists
                    if rt_ind > 2
                        post_reward_RTs_diff = [post_reward_RTs_diff; (RTs_currentBlock(rt_ind) - RTs_currentBlock(rt_ind-1) - (RTs_currentBlock(rt_ind-1)-RTs_currentBlock(rt_ind-2)))];
                    end
                    break;
                end
            end
        end
    end
    
    post_reward_RT_mean = mean(post_reward_RTs, 'omitnan'); % V
    post_reward_RT_std = std(post_reward_RTs, 'omitnan'); % V
    resp_time_diff_after_reward_mean = mean(post_reward_RTs_diff, 'omitnan'); % V
    resp_time_diff_after_reward_std = std(post_reward_RTs_diff, 'omitnan'); % V
    
    % compute burstiness:
    mu = mean(main_RT_table_run1.RT_diff, 'omitnan');
    sigma = std(main_RT_table_run1.RT_diff, 'omitnan');
    if mu + sigma == 0  % Avoid division by zero
        disp(['mu + sigma is zero for irt: ', num2str(main_RT_table_run1.RT_diff)])
        run_burstiness = NaN; % V
    else
        run_burstiness = (sigma - mu) / (sigma + mu); % V
    end
    
    % Assemble all features into a table
    run1_features = table;
    run1_features.oldSubID = str2num(subjX);
    run1_features.newID = str2num(newIDX);
    run1_features.group = groupName{1};
    run1_features.site = siteX;
    run1_features.day = 1;
    run1_features.run = 1;
    run1_features.resp_rate_trial_mean = resp_rate_trial_mean;
    run1_features.resp_rate_trial_std = resp_rate_trial_std;
    run1_features.resp_pairs_sum_std = resp_pairs_sum_std;
    run1_features.RTs_first_resp_mean = RTs_first_resp_mean;
    run1_features.RTs_first_resp_std = RTs_first_resp_std;
    run1_features.isCorrectResp_prop_mean = isCorrectResp_prop_mean;
    run1_features.isCorrectResp_prop_std = isCorrectResp_prop_std;
    run1_features.resp_pairs_sum_slope = resp_pairs_sum_slope;
    run1_features.resp_pairs_sum_spearman_corr = resp_pairs_sum_spearman_corr;
    run1_features.resp_time_diff_mean = resp_time_diff_mean;
    run1_features.resp_time_diff_std = resp_time_diff_std;
    run1_features.volatility = volatility;
    run1_features.right_vs_left = right_vs_left;
    run1_features.sweet_vs_salty = sweet_vs_salty;
    run1_features.post_reward_RT_mean = post_reward_RT_mean;
    run1_features.post_reward_RT_std = post_reward_RT_std;
    run1_features.resp_time_diff_after_reward_mean = resp_time_diff_after_reward_mean;
    run1_features.resp_time_diff_after_reward_std = resp_time_diff_after_reward_std;
    run1_features.run_burstiness = run_burstiness;
    
    % Append to main table
    run1_features_table = [run1_features_table; run1_features];
    % end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% get the kind of snack that has been devalued
    
end

%% print database
cd (databases_dir)
writetable(run1_features_table, [analysis_name '.csv']);