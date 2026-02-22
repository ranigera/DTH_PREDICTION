# Importing libraries
import os        
import time        
import numpy  as np  
import pandas as pd
from   subject import Subject

# Directories
PROJECT_DIRECTORY          = '..'
TRIAL_INFO                 = os.path.join(PROJECT_DIRECTORY,     'trial-info')   # Storing the subject-wise trial-information
CSV_DIRECTORY              = os.path.join(PROJECT_DIRECTORY,     'csvs')         # all the behavioral variable information

def get_str_subid(subid):
    """Generate the RO1 ID for a given subject ID."""
    if isinstance(subid, str):
        try:
            subid = int(subid.split('-')[1])  # Extract integer from string
        except (IndexError, ValueError):
            raise ValueError("Invalid subject ID format. Expected format: 'prefix-<number>'")
    return f'R01_{subid:04d}'

# Get the trial information for the participant
def get_trial_info(subid):
    return os.path.join(TRIAL_INFO, get_str_subid(int(subid)) + '-trial_info.csv')

# Get the bug-type
def get_bug(subid):
    if subid > 112: 
        return 'slowMouseNoBug'
    else: 
        df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'choiceTestDevaluationRatio.csv'))
        return df[df['participant'] == 'sub-'+str(subid)]['Bug'].iat[0]

# Get the contingencies (coin, stim) for the participant
def get_contingencies(subid):
    coin, stim, count = 0, 0, 0
    df = pd.read_csv(get_trial_info(subid))
    df = df[df['event'] == 'contingency_resp']
    for index, row in df.iterrows():
        if count < 2:
            coin += float(row['corrResp'] == row['resp'])
        else:
            stim += float(row['corrResp'] == row['resp'])
        count += 1
    return coin, stim

# Get the consumption score (score for consumption test out of 10)
def get_consumption_score(subid):
    df = pd.read_csv(get_trial_info(subid))
    df = df[df['run'] == 1]
    df = df[df['phase'] == 'consumption']
    devalued_coin = df['devalued_coin'].iat[0]
    score = 0.
    for each in list(df['selected_coin']):
        if each in ['silver', 'gold'] and each != devalued_coin:
            score += 1
    return score

# Get the devalued direction (left or right)
def get_devalued_direction(subid):
    df = pd.read_csv(get_trial_info(subid))
    return int(df[df['coin_img'] == 'stim/' + df['devalued_coin'].iat[0] + '_coin.png']['corrResp'].iat[0] == 'left')

# Get the devalued coin (gold or silver)
def get_devalued_coin(subid):
    return int(pd.read_csv(get_trial_info(subid))['devalued_coin'].iat[0].lower() == 'gold')

# Get the task-ordering (whether MBMF was first or Habit)
def get_task_ordering(subid):
    return int(pd.read_csv(get_trial_info(subid))['task_ordering'].iat[0]) - 1

# Get the handedness
def get_hand(subid):
    try:
        return int(pd.read_csv(get_trial_info(subid))['hand'].iat[0].lower() == 'right')
    except: 
        return np.nan

# estimating the devaluation ratio for the subid
def Estimate_devaluation_ratio(subid):
    df = pd.read_csv(get_trial_info(subid))
    temp = df[df['phase'] == 'training'].loc[0]
    if 'silver' in temp['coin_img']:
        if 'left' in temp['corrResp']:
            silver_response = 'left'
            gold_response   = 'right'
        else:
            silver_response = 'right'
            gold_response   = 'left'
    else:
        if 'left' in temp['corrResp']:
            silver_response = 'right'
            gold_response   = 'left'
        else:
            silver_response = 'left'
            gold_response   = 'right'
    if temp['devalued_coin'] == 'silver':
        correct = gold_response
    else:
        correct = silver_response
    temp = df[df['event'] == 'choice_resp']['resp']
    return (temp[temp == correct].size)/temp.shape[0]

# Get choice block
def get_choice_block(subid):
    df = pd.read_csv(get_trial_info(subid))         # getting the trial-information
    return df[df['event'].isin(['choice_test_start', 'choice_resp', 'instr_before_contingency_test'])] # getting the training-blocks


# Get the devaluation ratio for the subid
def get_devaluation_ratio(subid):
    if get_bug(subid) == 'slowMouseNoBug': return Estimate_devaluation_ratio(subid) # if there is no-bug, estimate from the trial-information
    # else, get from the csv file
    df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'choiceTest_devaluationRatio_updatedParticipantList.csv'))
    for index, row in df.iterrows():
        if subid == int(row['ID'].split('-')[1].replace('t', '').replace('_', '')):
            return float(row['devaluation_ratio'])

# Getting the over-all rate
def get_rate(subid):
    try:
        return np.mean([(get_block(subid, runid, blockid).shape[0] - 2.)/(get_block(subid, runid, blockid).iloc[-1]['globalClock_t'] - get_block(subid, runid, blockid).iloc[0]['globalClock_t']) for runid in [1, 2] for blockid in range(1, 11)])
    except:
        return np.nan

# Get the training block
def get_block(subid, runid, blockid):
    df = pd.read_csv(get_trial_info(subid))         # getting the trial-information
    df = df[df['run'] == runid - 1].reset_index()   # getting the information for the run
    temp = df[df['event'].isin(['training_period_start', 'ITI_start'])] # getting the training-blocks
    curr = 1    # current block
    skip = False    
    start = None
    for index, row in temp.iterrows():
        if skip and start is None: 
            skip=False
            continue 
        else:
            if start is None: 
                skip = True
            else:
                return df.iloc[start:index+1]
        if curr == blockid:
            start = index
        else: 
            curr += 1

def get_block_rate(subid, runid, blockid):
    return (get_block(subid, runid, blockid).shape[0] - 2.)/(get_block(subid, runid, blockid).iloc[-1]['globalClock_t'] - get_block(subid, runid, blockid).iloc[0]['globalClock_t'])


if __name__ == '__main__':
    subjects = Subject.get_all_subjects()
    print(get_trial_info(subjects[0].subid))