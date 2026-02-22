# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import csv
import json
import datetime
from datetime import date, datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os.path
from os import path, listdir
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import inspect
from scipy.ndimage import generic_filter
import pickle
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import statsmodels.api as sm
import scipy.stats as stats

# R stuff:
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
importr('glmmTMB')
importr('lme4')
pandas2ri.activate()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# %% [markdown]
# ## Exclusion and other notes:

# %%
# Excluded due to random gamble choices (pressing key many times before arrows even displayed):
Exclusions = [1056, 1081, 3005, 3021, 3091]

# Excluded due to what appears as a considerable lack of engagement:
# 3056 Exclude (did not really do the SRO part [just pressed many button again and again]])
# 3095 - responded in ~5% of SRO trials and ~66% of gambles. * Turned out in the messages that he/she completely misunderstoond the instructions.
Exclusions = Exclusions + [3056, 3095]

# * note on 3055: was timed out but asked to accept submission. He stopped in the middle once formore than 35 minutes and once for ~65 minutes, I did not approve. * I guess maybe not need to indicate in reports as they were timed out. * also by the way he had (74% in gambles 49% in SRO)
Exclusions = Exclusions + [3055]
Exclusions = Exclusions + []

# Other relevant info:
# 3058 and 3060 do not have the very last trial recorded and no backup data recorded, and no questionnaires; fortunately, they have all devaluation trials.
# 1055 - has good data but no questionnaires (reported that it refreshed or something between the task and questionnaires).
# The first subject with 2 failed attention checks in the questionnaires wrote to say that he spotted on of them but could no change it.

POTENTIAL_EXCLUSIONS = {}


# %% [markdown]
# # Functions and parameter definitions

# %%
# Save raw data as .json file
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def get_global_var_name(var):
    """
    Returns the name of a global variable as a string.
    """
    for name, value in globals().items():
        if value is var:
            return name

# %%
# Calculate local standard deviation:
# -------------------------------------------


def local_std(data, window_size=5):
    return generic_filter(data, np.std, size=window_size, mode='constant', cval=np.nan)
# Calculate rolling standard deviation:
# -------------------------------------------


def rolling_std(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).std()


# %% [markdown]
# # *------------- DATA READING FROM SAVED FILES ---------------*

# %% [markdown]
# ## READ Main DF and CT DF
# %%
main_data_df = pd.read_pickle('parsed_data/main_data_df.pkl')
CT_data = pd.read_pickle('parsed_data/CT_data.pkl')
gambles = pd.read_pickle('parsed_data/gambles.pkl')
questionnaire_scores = pd.read_csv('parsed_data/questionnaire_scores.csv')


# %% [markdown]
# ## Apply exclusion

# %%
print(Exclusions)
# apply exclusions according to Exclusions variable:
main_data_df = main_data_df[~main_data_df['sub'].isin(Exclusions)]
CT_data = CT_data[~CT_data['sub'].isin(Exclusions)]
gambles = gambles[~gambles['sub'].isin(Exclusions)]

main_data_df = main_data_df.reset_index(drop=True)
CT_data = CT_data.reset_index(drop=True)
gambles = gambles.reset_index(drop=True)

# %% [markdown]
# # Create other dfs of interest

# %% [markdown]
# ### Create other gamble DFs (from the main gambles df)

# %%
# Separate the gambles missed_gambles and no_miss_gambles:
missed_gambles = gambles[gambles['chosen_gamble'].isna()]
no_miss_gambles = gambles[~gambles['chosen_gamble'].isna()]

# ------------------------------------------------------------------------------
# Assemble EV ratio=1 gambles [excluding missed gambles]:
# ------------------------------------------------------------------------------
# get data wihtout trials with gamble_type of EV_ratio_1:
gambles_EV1ratio = no_miss_gambles[no_miss_gambles['gamble_type']
                                   == 'EV_ratio_1']
gambles_EV1ratio = gambles_EV1ratio.reset_index(drop=True)

# Test for choosing safer (higher chance gamble):
gambles_EV1ratio['isBottomSafer'] = gambles_EV1ratio.apply(
    lambda x: x['gambles_bottom'][1] > x['gambles_top'][1], axis=1)
gambles_EV1ratio['chose_bottom'] = gambles_EV1ratio.apply(
    lambda x: x['chosen_gamble'] == x['gambles_bottom'], axis=1)

# check if choice is safer than the other:
gambles_EV1ratio['chose_safer'] = gambles_EV1ratio['isBottomSafer'] == gambles_EV1ratio['chose_bottom']

# ------------------------------------------------------------------------------
# Assemble regular gambles (medium/easy/hard) [excluding missed gambles]:
# ------------------------------------------------------------------------------
# get data w/o trials with gamble_type of EV_ratio_1:
gambles_w_no_EV1 = no_miss_gambles[no_miss_gambles['gamble_type']
                                   != 'EV_ratio_1']
gambles_w_no_EV1 = gambles_w_no_EV1.reset_index(drop=True)

# Test for choosing higher EV gamble:
gambles_w_no_EV1['isBottomBetter'] = gambles_w_no_EV1.apply(
    lambda x: x['gambles_bottom'][0] * x['gambles_bottom'][1] > x['gambles_top'][0] * x['gambles_top'][1], axis=1)
gambles_w_no_EV1['chose_bottom'] = gambles_w_no_EV1.apply(
    lambda x: x['chosen_gamble'] == x['gambles_bottom'], axis=1)
# check if choice is better than the other:
gambles_w_no_EV1['chose_best_EV'] = gambles_w_no_EV1['isBottomBetter'] == gambles_w_no_EV1['chose_bottom']

# get the regular gambles:
regular_gambles = gambles_w_no_EV1[gambles_w_no_EV1['gamble_type']
                                   != 'sanityCheck'].copy()
# Test for choosing safer (higher chance gamble):
regular_gambles['isBottomSafer'] = regular_gambles.apply(
    lambda x: x['gambles_bottom'][1] > x['gambles_top'][1], axis=1)
# check if choice is safer than the other:
regular_gambles['chose_safer'] = regular_gambles['isBottomSafer'] == regular_gambles['chose_bottom']

# calculate the average number of chose_safer per sub per block per gamble_type:
# true and true, false and false EV prob higher
regular_gambles['higherElementInBestEVgambles'] = regular_gambles['isBottomBetter'] == regular_gambles['isBottomSafer']
regular_gambles['higherElementInBestEVgambles'] = regular_gambles.apply(
    lambda x: 'probability' if x['higherElementInBestEVgambles'] else 'gain', axis=1)

# ------------------------------------------------------------------------------
# Assemble sanity check gambles [excluding missed gambles]:
# ------------------------------------------------------------------------------
# now get only sainty check gambles:
sanityCheck_gambles = gambles_w_no_EV1[gambles_w_no_EV1['gamble_type'] == 'sanityCheck'].copy(
)
sanityCheck_gambles = sanityCheck_gambles.reset_index(drop=True)


# %% [markdown]
# ### Sequences entered in a row dataframe - and plotting functions

# %%
# Helper functions
# -----------------------------
def extractCorrect_CONSECUTIVE_KeyPresses(seq, presses):
    trial_key_presses = presses.copy()
    trial_keys_str = ''.join([x['key_pressed'] for x in trial_key_presses])
    first_good_consecutive_seq = trial_keys_str.find(seq)
    if first_good_consecutive_seq == -1:
        return []
    # this to overcome rare key appear with rt None and then again with actual rt
    if first_good_consecutive_seq + len(seq) - 1 < len(trial_key_presses) and trial_key_presses[first_good_consecutive_seq+len(seq)-1]['key_rt'] == None:
        return trial_key_presses[first_good_consecutive_seq:first_good_consecutive_seq+len(seq)+1]
    return trial_key_presses[first_good_consecutive_seq:first_good_consecutive_seq+len(seq)]


def calculateSeqCompletoinTime(presses):
    if len(presses) == 0:
        return np.nan
    return presses[-1]['key_rt'] - presses[0]['key_rt']


def calculateSeqCompletoinTimeFromStart(presses):
    if len(presses) == 0:
        return np.nan
    return presses[-1]['key_rt']


# Create the IPI data:
# -----------------------------
SeqsInRow_data = main_data_df[main_data_df['blockType']
                              != 'gambles_only'].copy()


# create a new column based on SRO_keyPressSummary that includes only the correct sequence pressing (according to stim_seq)
SeqsInRow_data.loc[:, 'SRO_keyPressSummary_correct'] = SeqsInRow_data.apply(
    lambda row: extractCorrect_CONSECUTIVE_KeyPresses(row['stim_seq'], row['SRO_keyPressSummary']), axis=1)
SeqsInRow_data.loc[:, 'SRO_seq_completion_time_SeqsInRow'] = SeqsInRow_data.apply(
    lambda row: calculateSeqCompletoinTime(row['SRO_keyPressSummary_correct']), axis=1)
SeqsInRow_data.loc[:, 'SRO_seq_completion_time_SeqsInRow_from_start'] = SeqsInRow_data.apply(
    lambda row: calculateSeqCompletoinTimeFromStart(row['SRO_keyPressSummary_correct']), axis=1)
# reset the index:
SeqsInRow_data.reset_index(inplace=True, drop=True)

# %% [markdown]
# ### IPI consistency data
# (see more details where I begin plotting it)

# %%
# Helper functions
# -----------------------------


def extractCorrectKeyPresses(seq, presses):
    trial_key_presses = presses.copy()
    # remove key presses woth key_rt = None:
    trial_key_presses = [
        x for x in trial_key_presses if x['key_rt'] is not None]
    correct_presses = []
    min_loc_to_look = 0  # used to verify the order of the keys.
    if trial_key_presses:
        for i in range(len(seq)):
            trial_key_presses = trial_key_presses[min_loc_to_look:]
            if seq[i] in [x['key_pressed'] for x in trial_key_presses]:
                loc = [x['key_pressed']
                       for x in trial_key_presses].index(seq[i])
                correct_presses.append(trial_key_presses[loc])
                min_loc_to_look = loc + 1
    return correct_presses


def calculateIPIs(key_presses):
    ipis = np.nan
    if len(key_presses) == 3:
        IPI1 = key_presses[1]['key_rt'] - key_presses[0]['key_rt']
        IPI2 = key_presses[2]['key_rt'] - key_presses[1]['key_rt']
        ipis = [IPI1, IPI2]
    return ipis


def calc_IPI_Consistency(input_data, seperate_by_block=False):
    # cerate a copy of the data
    data = input_data.copy()
    for i in range(1, len(data)):
        row1 = data.iloc[i-1]
        row2 = data.iloc[i]
        isSameBlock = row1['block'] == row2['block']
        if row1['sub'] == row2['sub'] and row1['stimType'] == row2['stimType'] and \
                isinstance(row1['inter_press_intervals'], list) and isinstance(row2['inter_press_intervals'], list) and \
                row1['inter_press_intervals'] and row2['inter_press_intervals']:
            if not seperate_by_block or isSameBlock:
                # display(row1['inter_press_intervals'])
                # display(row2['inter_press_intervals'])
                IPI_1_abs_diff = abs(
                    row2['inter_press_intervals'][0] - row1['inter_press_intervals'][0])
                IPI_2_abs_diff = abs(
                    row2['inter_press_intervals'][1] - row1['inter_press_intervals'][1])
                # calculate the sum of the absolute difference between the IPIs:
                IPI_abs_diff_sum = IPI_1_abs_diff + IPI_2_abs_diff

                # add it to the dataframe:
                data.loc[i, 'IPI_abs_diff_sum'] = IPI_abs_diff_sum
    return data


# Create the IPI data:
# -----------------------------
IPI_consistency_data = main_data_df[main_data_df['blockType']
                                    != 'gambles_only'].copy()

# create a new column based on SRO_keyPressSummary that includes only the correct sequence pressing (according to stim_seq)
IPI_consistency_data.loc[:, 'SRO_keyPressSummary_correct'] = IPI_consistency_data.apply(
    lambda row: extractCorrectKeyPresses(row['stim_seq'], row['SRO_keyPressSummary']), axis=1)
IPI_consistency_data.loc[:, 'inter_press_intervals'] = IPI_consistency_data.apply(
    lambda row: calculateIPIs(row['SRO_keyPressSummary_correct']), axis=1)

# Create a new dataframes with the IPIs (with and without block seperation)
sorted_IPI_consistency_data = IPI_consistency_data.sort_values(
    by=['sub', 'stimType', 'block', 'trial'])
sorted_IPI_consistency_data = sorted_IPI_consistency_data.reset_index(
    drop=True)

# add absolute stimulus trial number:
sorted_IPI_consistency_data.loc[:, 'stim_abs_trial'] = sorted_IPI_consistency_data.groupby(
    ['sub', 'stimType']).cumcount() + 1

IPI_consistency_data_by_trial = calc_IPI_Consistency(
    sorted_IPI_consistency_data, seperate_by_block=False)
IPI_consistency_data_w_block_sep = calc_IPI_Consistency(
    sorted_IPI_consistency_data, seperate_by_block=True)

# %% [markdown]
# ## Create SUMMARY DATA (of all kind of stuff)

# %% [markdown]
# ### volatility - RT
# I can also add regular std etc.

# %%
window_size = 5
# ----------------------------------------------------------------------------------------------------------------------
# clculate volatility based on the mean of the absolute difference between each trial and the previous one:
# ----------------------------------------------------------------------------------------------------------------------
volatility_data = main_data_df[(main_data_df['blockType'] == 'dual') & (main_data_df['stim_condition'] != 'never_valued') & (
    main_data_df['phase'] != 'test') & (main_data_df['phase'] != 'reacquisition')].copy()
volatility_data = volatility_data.reset_index(drop=True)
volatility_data = volatility_data[volatility_data['SRO_rt_of_SRO_key'].notna()]
volatility_data.loc[:, 'abs_diff_SRO_rt_of_SRO_key'] = volatility_data.groupby(
    ['sub', 'group', 'stim_condition', 'block'])['SRO_rt_of_SRO_key'].diff()
# get the absolute value of the diff:
volatility_data.loc[:,
                    'abs_diff_SRO_rt_of_SRO_key'] = volatility_data['abs_diff_SRO_rt_of_SRO_key'].abs()
SUMMARY_volatility_data_calcSeperatelyPerStim = volatility_data.groupby(
    ['sub', 'group'])['abs_diff_SRO_rt_of_SRO_key'].mean().reset_index()
SUMMARY_volatility_data_calcSeperatelyPerStim = SUMMARY_volatility_data_calcSeperatelyPerStim.rename(
    columns={'abs_diff_SRO_rt_of_SRO_key': 'volatility_mean_abs_diff_SRO_rt_of_SRO_key_calcSeperatelyPerStim'})

# now do the same but without seperating by stim_condition:
volatility_data = main_data_df[(main_data_df['blockType'] == 'dual') & (main_data_df['stim_condition'] != 'never_valued') & (
    main_data_df['phase'] != 'test') & (main_data_df['phase'] != 'reacquisition')].copy()
volatility_data = volatility_data.reset_index(drop=True)
volatility_data = volatility_data[volatility_data['SRO_rt_of_SRO_key'].notna()]
volatility_data.loc[:, 'abs_diff_SRO_rt_of_SRO_key'] = volatility_data.groupby(
    ['sub', 'group', 'block'])['SRO_rt_of_SRO_key'].diff()
# get the absolute value of the diff:
volatility_data.loc[:,
                    'abs_diff_SRO_rt_of_SRO_key'] = volatility_data['abs_diff_SRO_rt_of_SRO_key'].abs()
SUMMARY_volatility_data = volatility_data.groupby(
    ['sub', 'group'])['abs_diff_SRO_rt_of_SRO_key'].mean().reset_index()
SUMMARY_volatility_data = SUMMARY_volatility_data.rename(
    columns={'abs_diff_SRO_rt_of_SRO_key': 'volatility_mean_abs_diff_SRO_rt_of_SRO_key'})

# Calculate rolling standard deviation for each group, sub, phase, stim_condition
volatility_data = main_data_df[(main_data_df['blockType'] == 'dual') & (main_data_df['stim_condition'] != 'never_valued') & (
    main_data_df['phase'] != 'test') & (main_data_df['phase'] != 'reacquisition')].copy()
volatility_data = volatility_data.reset_index(drop=True)
volatility_data = volatility_data[volatility_data['SRO_rt_of_SRO_key'].notna()]
volatility_data = volatility_data.groupby(['sub', 'group', 'stim_condition', 'block'])[
    'SRO_rt_of_SRO_key'].apply(lambda x: rolling_std(x)).reset_index()
volatility_data = volatility_data.rename(
    columns={'SRO_rt_of_SRO_key': 'rolling_std_SRO_rt_of_SRO_key_calcSeperatelyPerStim'})
SUMMARY_volatility_data_rolling_std_calcSeperatelyPerStim = volatility_data.groupby(
    ['sub', 'group'])['rolling_std_SRO_rt_of_SRO_key_calcSeperatelyPerStim'].mean().reset_index()

# now do the same but without seperating by stim_condition:
volatility_data = main_data_df[(main_data_df['blockType'] == 'dual') & (main_data_df['stim_condition'] != 'never_valued') & (
    main_data_df['phase'] != 'test') & (main_data_df['phase'] != 'reacquisition')].copy()
volatility_data = volatility_data.reset_index(drop=True)
volatility_data = volatility_data[volatility_data['SRO_rt_of_SRO_key'].notna()]
volatility_data = volatility_data.groupby(['sub', 'group', 'block'])[
    'SRO_rt_of_SRO_key'].apply(lambda x: rolling_std(x)).reset_index()
volatility_data = volatility_data.rename(
    columns={'SRO_rt_of_SRO_key': 'rolling_std_SRO_rt_of_SRO_key'})
SUMMARY_volatility_data_rolling_std = volatility_data.groupby(
    ['sub', 'group'])['rolling_std_SRO_rt_of_SRO_key'].mean().reset_index()

# integrate all dataframes:
SUMMARY_volatility_data = SUMMARY_volatility_data.merge(
    SUMMARY_volatility_data_calcSeperatelyPerStim, on=['sub', 'group'])
SUMMARY_volatility_data = SUMMARY_volatility_data.merge(
    SUMMARY_volatility_data_rolling_std, on=['sub', 'group'])
SUMMARY_volatility_data = SUMMARY_volatility_data.merge(
    SUMMARY_volatility_data_rolling_std_calcSeperatelyPerStim, on=['sub', 'group'])

# %% [markdown]
# ## Gamble choice bias (button and location)

# %%
chosen_left_all = main_data_df.groupby(['group', 'sub'])['chosen_direction'].apply(
    lambda x: (x == 'left').sum()/(x.isin(['right', 'left'])).sum()).reset_index()
chosen_left_all.columns = ['group', 'sub', 'chose_left']

POTENTIAL_EXCLUSIONS['more_than_80percent_same_button_gambles_over_all'] = sorted(
    chosen_left_all.loc[np.abs(chosen_left_all['chose_left'] - 0.5) >= 0.3, :]['sub'].unique().tolist())

# plot the proportion of chosen_direction == left:
chosen_left = main_data_df.groupby(['group', 'sub', 'block'])['chosen_direction'].apply(
    lambda x: (x == 'left').sum()/(x.isin(['right', 'left'])).sum()).reset_index()
chosen_left.columns = ['group', 'sub', 'block', 'chose_left']

POTENTIAL_EXCLUSIONS['more_than_90percent_same_button_gambles_in_at_least_one_block'] = sorted(
    chosen_left.loc[np.abs(chosen_left['chose_left'] - 0.5) >= 0.4, :]['sub'].unique().tolist())

chosen_left_by_phase = main_data_df.groupby(['group', 'sub', 'block', 'phase'])['chosen_direction'].apply(
    lambda x: (x == 'left').sum()/(x.isin(['right', 'left'])).sum()).reset_index()
POTENTIAL_EXCLUSIONS['more_than_80percent_same_button_gambles_in_test_block'] = sorted(chosen_left_by_phase.loc[(
    chosen_left_by_phase['phase'] == 'test') & (np.abs(chosen_left_by_phase['chosen_direction'] - 0.5) >= 0.3), :]['sub'].unique().tolist())
POTENTIAL_EXCLUSIONS['more_than_80percent_same_button_gambles_in_reacquisition_block'] = sorted(chosen_left_by_phase.loc[(
    chosen_left_by_phase['phase'] == 'reacquisition') & (np.abs(chosen_left_by_phase['chosen_direction'] - 0.5) >= 0.3), :]['sub'].unique().tolist())

chosen_bottom_all = main_data_df.groupby(['group', 'sub'])['chosen_location'].apply(lambda x: (
    x == 'arrow_bottom').sum()/(x.isin(['arrow_bottom', 'arrow_top'])).sum()).reset_index()
chosen_bottom_all.columns = ['group', 'sub', 'chosen_bottom']

POTENTIAL_EXCLUSIONS['more_than_80percent_same_location_gambles_over_all'] = sorted(
    chosen_bottom_all.loc[np.abs(chosen_bottom_all['chosen_bottom'] - 0.5) >= 0.3, :]['sub'].unique().tolist())

# plot the proportion of chosen_direction == left:
chosen_bottom = main_data_df.groupby(['group', 'sub', 'block'])['chosen_location'].apply(
    lambda x: (x == 'arrow_bottom').sum()/(x.isin(['arrow_bottom', 'arrow_top'])).sum()).reset_index()
chosen_bottom.columns = ['group', 'sub', 'block', 'chosen_bottom']

POTENTIAL_EXCLUSIONS['more_than_90percent_same_location_gambles_in_at_least_one_block'] = sorted(
    chosen_bottom.loc[np.abs(chosen_bottom['chosen_bottom'] - 0.5) >= 0.4, :]['sub'].unique().tolist())

chosen_bottom_by_phase = main_data_df.groupby(['group', 'sub', 'block', 'phase'])['chosen_location'].apply(
    lambda x: (x == 'arrow_bottom').sum()/(x.isin(['arrow_bottom', 'arrow_top'])).sum()).reset_index()
POTENTIAL_EXCLUSIONS['more_than_80percent_same_location_gambles_in_test_block'] = sorted(chosen_bottom_by_phase.loc[(
    chosen_bottom_by_phase['phase'] == 'test') & (np.abs(chosen_bottom_by_phase['chosen_location'] - 0.5) >= 0.3), :]['sub'].unique().tolist())
POTENTIAL_EXCLUSIONS['more_than_80percent_same_location_gambles_in_reacquisition_block'] = sorted(chosen_bottom_by_phase.loc[(
    chosen_bottom_by_phase['phase'] == 'reacquisition') & (np.abs(chosen_bottom_by_phase['chosen_location'] - 0.5) >= 0.3), :]['sub'].unique().tolist())


chosen_stay_all = main_data_df.groupby(['group', 'sub'])['stay_switch_button'].apply(
    lambda x: (x == 'stay').sum()/(x.isin(['stay', 'switch'])).sum()).reset_index()
chosen_stay_all.columns = ['group', 'sub', 'stay_switch_button']
POTENTIAL_EXCLUSIONS['more_than_80percent_STAY_or_SWITCH_button_gambles_over_all'] = sorted(
    chosen_stay_all.loc[np.abs(chosen_stay_all['stay_switch_button'] - 0.5) >= 0.3, :]['sub'].unique().tolist())

chosen_stay = main_data_df.groupby(['group', 'sub', 'block'])['stay_switch_button'].apply(
    lambda x: (x == 'stay').sum()/(x.isin(['stay', 'switch'])).sum()).reset_index()
chosen_stay.columns = ['group', 'sub', 'block', 'stay_switch_button']
POTENTIAL_EXCLUSIONS['more_than_90percent_STAY_or_SWITCH_button_gambles_in_at_least_one_block'] = sorted(
    chosen_stay.loc[np.abs(chosen_stay['stay_switch_button'] - 0.5) >= 0.4, :]['sub'].unique().tolist())

chosen_stay_by_phase = main_data_df.groupby(['group', 'sub', 'block', 'phase'])['stay_switch_button'].apply(
    lambda x: (x == 'stay').sum()/(x.isin(['stay', 'switch'])).sum()).reset_index()
POTENTIAL_EXCLUSIONS['more_than_80percent_STAY_or_SWITCH_button_gambles_in_test_block'] = sorted(chosen_stay_by_phase.loc[(
    chosen_stay_by_phase['phase'] == 'test') & (np.abs(chosen_stay_by_phase['stay_switch_button'] - 0.5) >= 0.3), :]['sub'].unique().tolist())
POTENTIAL_EXCLUSIONS['more_than_80percent_STAY_or_SWITCH_button_gambles_in_reacquisition_block'] = sorted(chosen_stay_by_phase.loc[(
    chosen_stay_by_phase['phase'] == 'reacquisition') & (np.abs(chosen_stay_by_phase['stay_switch_button'] - 0.5) >= 0.3), :]['sub'].unique().tolist())


chosen_stay_LOC_all = main_data_df.groupby(['group', 'sub'])['stay_switch_location'].apply(
    lambda x: (x == 'stay').sum()/(x.isin(['stay', 'switch'])).sum()).reset_index()
chosen_stay_LOC_all.columns = ['group', 'sub', 'stay_switch_location']

POTENTIAL_EXCLUSIONS['more_than_80percent_STAY_or_SWITCH_location_gambles_over_all'] = sorted(
    chosen_stay_LOC_all.loc[np.abs(chosen_stay_LOC_all['stay_switch_location'] - 0.5) >= 0.3, :]['sub'].unique().tolist())

chosen_stay_LOC = main_data_df.groupby(['group', 'sub', 'block'])['stay_switch_location'].apply(
    lambda x: (x == 'stay').sum()/(x.isin(['stay', 'switch'])).sum()).reset_index()
chosen_stay_LOC.columns = ['group', 'sub', 'block', 'stay_switch_location']
POTENTIAL_EXCLUSIONS['more_than_90percent_STAY_or_SWITCH_location_gambles_in_at_least_one_block'] = sorted(
    chosen_stay_LOC.loc[np.abs(chosen_stay_LOC['stay_switch_location'] - 0.5) >= 0.4, :]['sub'].unique().tolist())

chosen_stay_LOC_by_phase = main_data_df.groupby(['group', 'sub', 'block', 'phase'])['stay_switch_location'].apply(
    lambda x: (x == 'stay').sum()/(x.isin(['stay', 'switch'])).sum()).reset_index()
POTENTIAL_EXCLUSIONS['more_than_80percent_STAY_or_SWITCH_location_gambles_in_test_block'] = sorted(chosen_stay_LOC_by_phase.loc[(
    chosen_stay_LOC_by_phase['phase'] == 'test') & (np.abs(chosen_stay_LOC_by_phase['stay_switch_location'] - 0.5) >= 0.3), :]['sub'].unique().tolist())
POTENTIAL_EXCLUSIONS['more_than_80percent_STAY_or_SWITCH_location_gambles_in_reacquisition_block'] = sorted(chosen_stay_LOC_by_phase.loc[(
    chosen_stay_LOC_by_phase['phase'] == 'reacquisition') & (np.abs(chosen_stay_LOC_by_phase['stay_switch_location'] - 0.5) >= 0.3), :]['sub'].unique().tolist())

# add print row limit:
never_valued_sequences_blocks = main_data_df[~main_data_df['rewardType'].isin(['blue', 'red'])].groupby(
    ['group', 'sub', 'block', 'rewardType'])['sequenceCompleted'].sum().reset_index()
POTENTIAL_EXCLUSIONS['more_than_4_NEVER_VALUED_seq_completed_in_at_least_one_block'] = sorted(
    never_valued_sequences_blocks.loc[never_valued_sequences_blocks['sequenceCompleted'] >= 5, :]['sub'].unique().tolist())

never_valued_sequences_all = main_data_df[~main_data_df['rewardType'].isin(['blue', 'red'])].groupby(
    ['group', 'sub', 'rewardType'])['sequenceCompleted'].mean().reset_index()
POTENTIAL_EXCLUSIONS['more_than_14_percent_NEVER_VALUED_seq_completed_over_all'] = sorted(
    never_valued_sequences_all.loc[never_valued_sequences_all['sequenceCompleted'] >= 0.14, :]['sub'].unique().tolist())
POTENTIAL_EXCLUSIONS['more_than_20_percent_NEVER_VALUED_seq_completed_over_all'] = sorted(
    never_valued_sequences_all.loc[never_valued_sequences_all['sequenceCompleted'] >= 0.20, :]['sub'].unique().tolist())
sorted(never_valued_sequences_all.loc[never_valued_sequences_all['sequenceCompleted']
       >= 0.20, :]['sub'].unique().tolist())

# now for phase:
never_valued_sequences_phase = main_data_df[~main_data_df['rewardType'].isin(['blue', 'red'])].groupby(
    ['group', 'sub', 'phase', 'rewardType'])['sequenceCompleted'].sum().reset_index()
POTENTIAL_EXCLUSIONS['more_than_4_NEVER_VALUED_seq_completed_in_test'] = sorted(never_valued_sequences_phase.loc[(
    never_valued_sequences_phase['phase'] == 'test') & (never_valued_sequences_phase['sequenceCompleted'] >= 5), :]['sub'].unique().tolist())
# sorted(never_valued_sequences_phase.loc[(never_valued_sequences_phase['phase'] == 'test') & (never_valued_sequences_phase['sequenceCompleted'] >= 5),:]['sub'].unique().tolist())
POTENTIAL_EXCLUSIONS['more_than_4_NEVER_VALUED_seq_completed_in_reacquisition'] = sorted(never_valued_sequences_phase.loc[(
    never_valued_sequences_phase['phase'] == 'reacquisition') & (never_valued_sequences_phase['sequenceCompleted'] >= 5), :]['sub'].unique().tolist())
# sorted(never_valued_sequences_phase.loc[(never_valued_sequences_phase['phase'] == 'reacquisition') & (never_valued_sequences_phase['sequenceCompleted'] >= 5),:]['sub'].unique().tolist())


POTENTIAL_EXCLUSIONS['CT_non_manip_minus_manip_lt_6_DF'] = CT_data[CT_data['non_manip_minus_manip'] < 6]

CT_data[CT_data['non_manip_minus_manip'] < 6]


# %% [markdown]
# # ANALYSIS
#

# %% [markdown]
# ## Learning indices

# %% [markdown]
# ## Functions (for RT and Sequence completion):

# %%
def plotTimeVar(data, var_of_interest='SRO_seq_completion_time', time_var='block', var_of_comparison='rewardType', include_never_valued=False, hue_order=['blue', 'red'], pallette=['blue', 'red'], statistic='mean', include_test_pahse=True, y_label=None):
    if time_var == 'phase':
        data.loc[:, time_var] = pd.Categorical(data[time_var], categories=[
                                               'pre_test', 'test', 'reacquisition'], ordered=True)

    # get only trials with a reward type blue or red:
    if include_never_valued:
        relevant_data = data[data['rewardType'].isin(['blue', 'red', 'rock'])]
    else:
        relevant_data = data[data['rewardType'].isin(['blue', 'red'])]
    # remove trials with seq completion, i.e. there is no nan in the var_of_interest column:
    relevant_data = relevant_data[~relevant_data[var_of_interest].isna()].reset_index(drop=True)
    if not include_test_pahse:
        relevant_data = relevant_data[relevant_data['phase'] != 'test']
        relevant_data = relevant_data[relevant_data['phase'] != 'reacquisition'].reset_index(drop=True)
    # calclulate the relevant measure:
    if statistic == 'mean':
        # calculate the average seq completion time per sub prer block per reward type:
        relevant_data_summary = relevant_data.groupby(
            ['group', 'sub', time_var, var_of_comparison])[var_of_interest].mean()
    elif statistic == 'STD':
        # calculate the variance of seq completion time per sub prer block per reward type:
        relevant_data_summary = relevant_data.groupby(
            ['group', 'sub', time_var, var_of_comparison])[var_of_interest].std()

    relevant_data_summary = relevant_data_summary.reset_index()

    # Separate the data for the "short" group
    short_group = relevant_data_summary[relevant_data_summary['group'] == 'short']
    blockNumbers_short = short_group[time_var].unique()
    x_ticks_loc_short = [x for x in range(len(blockNumbers_short))]
    short_group.loc[:, time_var] = short_group[time_var].replace(
        blockNumbers_short, x_ticks_loc_short)

    # Separate the data for the "more_extensive" group
    more_extensive_group = relevant_data_summary[relevant_data_summary['group']
                                                 == 'more_extensive']
    blockNumbers_more_extensive = more_extensive_group[time_var].unique()
    x_ticks_loc_more_extensive = [
        x for x in range(len(blockNumbers_more_extensive))]
    more_extensive_group.loc[:, time_var] = more_extensive_group[time_var].replace(
        blockNumbers_more_extensive, x_ticks_loc_more_extensive)

    # Plotting
    # -----------------------------
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Plot for the "short" group
    sns.lineplot(data=short_group, x=time_var, y=var_of_interest, hue=var_of_comparison,
                 hue_order=hue_order, palette=pallette, ax=axes[0], legend=False, errorbar=('se', 1))
    sns.lineplot(data=short_group, x=time_var, y=var_of_interest, hue=var_of_comparison, hue_order=hue_order,
                 style='sub', markers=True, dashes=False, alpha=0.01, legend=False, palette=pallette, ax=axes[0])
    axes[0].set_title('Short training')
    axes[0].set_xlabel(time_var)
    if y_label is not None:
        axes[0].set_ylabel(y_label)
    else:
        axes[0].set_ylabel(f'{statistic} {var_of_interest}')
    axes[0].set_xticks(x_ticks_loc_short)
    axes[0].set_xticklabels(blockNumbers_short)

    # Plot for the "more_extensive" group
    sns.lineplot(data=more_extensive_group, x=time_var, y=var_of_interest, hue=var_of_comparison,
                 hue_order=hue_order, palette=pallette, ax=axes[1], errorbar=('se', 1))
    sns.lineplot(data=more_extensive_group, x=time_var, y=var_of_interest, hue=var_of_comparison, hue_order=hue_order,
                 style='sub', markers=True, dashes=False, alpha=0.01, legend=False, palette=pallette, ax=axes[1])
    axes[1].set_title('Extensive training')
    axes[1].set_xlabel(time_var)
    axes[1].set_xticks(x_ticks_loc_more_extensive)
    axes[1].set_xticklabels(blockNumbers_more_extensive)

    # plt.tight_layout()
    plt.show()


def plotTimeVarViolineAndBar(data, var_of_interest='SRO_seq_completion_time', time_var='block', var_of_comparison='rewardType', include_never_valued=False, x_order=['pre_test', 'test', 'reacquisition'], hue_order=['blue', 'red'], pallette=['blue', 'red'], statistic='mean', y_label=None, x_label=None):
    if time_var == 'phase':
        data.loc[:, time_var] = pd.Categorical(data[time_var], categories=[
                                               'pre_test', 'test', 'reacquisition'], ordered=True)

    # get only trials with a reward type blue or red:
    if include_never_valued:
        relevant_data = data[data['rewardType'].isin(['blue', 'red', 'rock'])]
    else:
        relevant_data = data[data['rewardType'].isin(['blue', 'red'])]
    # remove trials with seq completion, i.e. there is no nan in the var_of_interest column:
    relevant_data = relevant_data[~relevant_data[var_of_interest].isna()]
    # calclulate the relevant measure:
    if statistic == 'mean':
        # calculate the average seq completion time per sub prer block per reward type:
        relevant_data_summary = relevant_data.groupby(
            ['group', 'sub', time_var, var_of_comparison])[var_of_interest].mean()
    elif statistic == 'STD':
        # calculate the variance of seq completion time per sub prer block per reward type:
        relevant_data_summary = relevant_data.groupby(
            ['group', 'sub', time_var, var_of_comparison])[var_of_interest].std()

    relevant_data_summary = relevant_data_summary.reset_index()

    # Separate the data for the "short" group
    short_group = relevant_data_summary[relevant_data_summary['group'] == 'short']
    # blockNumbers_short = short_group[time_var].unique()
    # x_ticks_loc_short = [x for x in range(len(blockNumbers_short))]
    # short_group.loc[:, time_var] = short_group[time_var].replace(blockNumbers_short, x_ticks_loc_short)

    # Separate the data for the "more_extensive" group
    more_extensive_group = relevant_data_summary[relevant_data_summary['group']
                                                 == 'more_extensive']
    # blockNumbers_more_extensive = more_extensive_group[time_var].unique()
    # x_ticks_loc_more_extensive = [x for x in range(len(blockNumbers_more_extensive))]
    # more_extensive_group.loc[:, time_var] = more_extensive_group[time_var].replace(blockNumbers_more_extensive, x_ticks_loc_more_extensive)

    # Plotting
    # -----------------------------
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot for the "short" group
    if len(hue_order) == 2:
        sns.violinplot(data=short_group, x=time_var, y=var_of_interest, hue=var_of_comparison, order=x_order,
                       hue_order=hue_order, palette=pallette, ax=axes[0], split=True, gap=0, inner="quart")
    else:
        sns.violinplot(data=short_group, x=time_var, y=var_of_interest, hue=var_of_comparison, order=x_order,
                       hue_order=hue_order, palette=pallette, ax=axes[0], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    plt.setp(axes[0].collections, alpha=.3)

    axes[0].set_title('Short training')
    axes[0].set_xlabel(time_var)
    axes[0].set_ylabel(f'{statistic} {var_of_interest}')
    # axes[0].set_xticks(x_ticks_loc_short)
    # axes[0].set_xticklabels(blockNumbers_short)

    # Plot for the "more_extensive" group
    if len(hue_order) == 2:
        sns.violinplot(data=more_extensive_group, x=time_var, y=var_of_interest, hue=var_of_comparison,
                       order=x_order, hue_order=hue_order, palette=pallette, ax=axes[1], split=True, gap=0, inner="quart")
    else:
        sns.violinplot(data=more_extensive_group, x=time_var, y=var_of_interest, hue=var_of_comparison, order=x_order,
                       hue_order=hue_order, palette=pallette, ax=axes[1], inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    plt.setp(axes[1].collections, alpha=.3)
    axes[1].set_title('Extensive training')
    axes[1].set_xlabel(time_var)
    # axes[1].set_xticks(x_ticks_loc_more_extensive)
    # axes[1].set_xticklabels(blockNumbers_more_extensive)
    axes[1].set(ylabel=None)

    # plt.tight_layout()
    plt.show()

    # Plotting
    # -----------------------------
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot for the "short" group
    sns.barplot(data=short_group, x=time_var, y=var_of_interest, hue=var_of_comparison, order=x_order,
                hue_order=hue_order, palette=pallette, ax=axes[0], alpha=0.5, errorbar=('se', 1))

    axes[0].set_title('Short training')
    axes[0].set_xlabel(time_var)
    if y_label is not None:
        axes[0].set_ylabel(y_label)
    else:
        axes[0].set_ylabel(f'{statistic} {var_of_interest}')

    if x_label is not None:
        axes[0].set_xlabel(x_label)
    else:
        axes[0].set_xlabel(time_var)
    # axes[0].set_xticks(x_ticks_loc_short)
    # axes[0].set_xticklabels(blockNumbers_short)

    # Plot for the "more_extensive" group
    sns.barplot(data=more_extensive_group, x=time_var, y=var_of_interest, hue=var_of_comparison,
                order=x_order, hue_order=hue_order, palette=pallette, ax=axes[1], alpha=0.5, errorbar=('se', 1))
    axes[1].set_title('Extensive training')
    axes[1].set_xlabel(time_var)
    # axes[1].set_xticks(x_ticks_loc_more_extensive)
    # axes[1].set_xticklabels(blockNumbers_more_extensive)
    axes[1].set(ylabel=None)
    if x_label is not None:
        axes[1].set_xlabel(x_label)
    else:
        axes[1].set_xlabel(time_var)

    # plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Functions (for gamble measure by stim consition):

# %%

def plot_gamble_var_by_SRO_stim_condition(data, var, combineConditions=False):
    choice_rt_by_SRO_stim_condition = data.groupby(
        ['group', 'sub', 'phase', 'stim_condition'])[var].mean().reset_index()
    choice_rt_by_SRO_stim_condition['phase'] = pd.Categorical(
        choice_rt_by_SRO_stim_condition['phase'], categories=['pre_test', 'test', 'reacquisition'], ordered=True)

    # create a figure with 3 subplots:
    fig, axs = plt.subplots(1, 2, figsize=(21, 6))

    # make a violin plot for each group:
    for i, group in enumerate(['short', 'more_extensive']):
        # plot the data for the 'short' group in the first subplot:
        group_data = choice_rt_by_SRO_stim_condition[choice_rt_by_SRO_stim_condition.group == group]
        # sns.lineplot(data=group_data, x='phase', y=var, hue='stim_condition',  hue_order=['still_valued', 'devalued', 'never_valued'], ax=axs[i])
        # now do a barplot for each group:
        if combineConditions:
            sns.barplot(data=group_data, x='phase', hue='phase',
                        y=var, ax=axs[i], alpha=0.6, errorbar=('se', 1))
        else:
            sns.barplot(data=group_data, x='phase', y=var, hue='stim_condition',  hue_order=[
                        'still_valued', 'devalued', 'never_valued'], ax=axs[i], alpha=0.6, errorbar=('se', 1))
        # make the first latter in group name capital:
        axs[i].set_title(f'{group[0].upper() + group[1:]} training')
        # make sure the y axis is the same for all subplots, accrding to the sublot with the largest y axis:
        axs[i].set_ylim(axs[0].get_ylim()) if axs[0].get_ylim()[1] > axs[i].get_ylim()[
            1] else axs[0].set_ylim(axs[i].get_ylim())


def plot_gamble_var_by_SRO_stim_condition_STD(data, var):
    choice_rt_by_SRO_stim_condition = data.groupby(
        ['group', 'sub', 'phase', 'stim_condition'])[var].std().reset_index()
    # remove sub 3005 and 3021:
    choice_rt_by_SRO_stim_condition = choice_rt_by_SRO_stim_condition[(
        choice_rt_by_SRO_stim_condition['sub'] != 3005) & (choice_rt_by_SRO_stim_condition['sub'] != 3021)]

    choice_rt_by_SRO_stim_condition['phase'] = pd.Categorical(
        choice_rt_by_SRO_stim_condition['phase'], categories=['pre_test', 'test', 'reacquisition'], ordered=True)

    # create a figure with 3 subplots:
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))

    # make a violin plot for each group:
    for i, group in enumerate(['short', 'more_extensive']):
        # plot the data for the 'short' group in the first subplot:
        group_data = choice_rt_by_SRO_stim_condition[choice_rt_by_SRO_stim_condition.group == group]
        # sns.lineplot(data=group_data, x='phase', y=var, hue='stim_condition',  hue_order=['still_valued', 'devalued', 'never_valued'], ax=axs[i])
        # now do a barplot for each group:
        sns.barplot(data=group_data, x='phase', y=var, hue='stim_condition',  hue_order=[
                    'still_valued', 'devalued', 'never_valued'], ax=axs[i], alpha=0.4, errorbar=('se', 1))
        # make the first latter in group name capital:
        axs[i].set_title(f'{group[0].upper() + group[1:]} training - STD')
        # make sure the y axis is the same for all subplots, accrding to the sublot with the largest y axis:
        axs[i].set_ylim(axs[0].get_ylim()) if axs[0].get_ylim()[1] > axs[i].get_ylim()[
            1] else axs[0].set_ylim(axs[i].get_ylim())


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# add time befroe/after (deval)
main_data_df['time'] = np.nan
main_data_df.loc[main_data_df['phase'] == 'pre_test', 'time'] = 'before'
main_data_df.loc[main_data_df['phase'] == 'test', 'time'] = 'after'
main_data_df.loc[main_data_df['phase'] == 'reacquisition', 'time'] = 'after'
main_data_df.loc[:, ['sub', 'phase', 'time']].tail(100)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

main_data_df = main_data_df.reset_index(drop=True)
CT_data = CT_data.reset_index(drop=True)
gambles = gambles.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Subjective utility related modelling additions and calculation subjective utility diff in each gamble etc.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ****** Based on having the file risk_modeling_results_powModel_TEMP.pkl fromed by the script 999_Risk_n_utility_n_choice_RT.ipynb ******

# now create a utility of the other way to calculate the utility function for risk aversion, using power utility

def utility_power(x, rho):
    return x**rho

# function to calculate accuracies:


def calculate_accuracy(gamble_data, fitted_data, utility_function):
    # rest index for fitted_data
    fitted_data.reset_index(drop=True, inplace=True)
    all_data = gamble_data.copy()

    # iterate rows in fitted_data and calculate accuracy for each subject
    for i, row in fitted_data.iterrows():
        # subject/line model data:
        subj = row['sub']
        rho = row.rho
        b = row.b
        # subject gamble data:
        data = all_data[all_data['sub'] == subj]
        # remove lines with missing data:
        data = data.dropna(subset=['choice_rt'])

        # Get the choices, probabilities, and payoffs
        choices = data['chose_bottom'].apply(lambda x: 1 if x == True else 0)
        p1 = data['gambles_bottom'].apply(lambda x: x[1])
        x1 = data['gambles_bottom'].apply(lambda x: x[0])
        p2 = data['gambles_top'].apply(lambda x: x[1])
        x2 = data['gambles_top'].apply(lambda x: x[0])

        # Calculate the expected utility of each gamble
        EU1 = p1 * utility_function(x1, rho)
        EU2 = p2 * utility_function(x2, rho)

        # Calculate the model's predicted choices
        prob_choice1 = 1 / (1 + np.exp(-b*(EU1-EU2)))
        predicted_choices = prob_choice1.apply(lambda x: 1 if x >= 0.5 else 0)

        # Calculate the accuracy as the proportion of choices that the model correctly predicts
        accuracy = np.mean(choices == predicted_choices)

        # add the accuracy to the fitted_data df
        fitted_data.loc[i, 'accuracy'] = accuracy


# test if the risk modelling file risk_modeling_results_powModel_TEMP.pkl exists and load it:
if os.path.exists('risk_modeling_results_powModel_TEMP.pkl'):
    risk_modeling_results = pd.read_pickle(
        'risk_modeling_results_powModel_TEMP.pkl')

    # get subject fitted rho and b values:
    long_df_rho = risk_modeling_results.melt(id_vars=['sub', 'group'], value_vars=[
                                             'rho_all_training_powModel'], var_name='block', value_name='rho')
    long_df_rho.sort_values(by=['sub'], inplace=True)
    long_df_rho['block'] = long_df_rho['block'].apply(
        lambda x: '_'.join(x.split('_')[1:]))

    long_df_b = risk_modeling_results.melt(id_vars=['sub', 'group'], value_vars=[
                                           'b_all_training_powModel'], var_name='block', value_name='b')
    long_df_b.sort_values(by=['sub'], inplace=True)
    long_df_b['block'] = long_df_b['block'].apply(
        lambda x: '_'.join(x.split('_')[1:]))

    long_df = pd.merge(long_df_rho, long_df_b, on=[
                       'sub', 'group', 'block'], how='outer')

    # now for each sub count the number of sequenceCompleted==1 for time == after and stim_condition == devalued:
    slip_seqs = main_data_df[(main_data_df['sequenceCompleted'] == 1) & (main_data_df['time'] == 'after') & (
        main_data_df['stim_condition'] == 'devalued')].groupby('sub').sequenceCompleted.count().reset_index()
    slip_press = main_data_df[(main_data_df['SRO_rt_of_SRO_key'].notna()) & (main_data_df['time'] == 'after') & (
        main_data_df['stim_condition'] == 'devalued')].groupby('sub').SRO_rt_of_SRO_key.count().reset_index()
    # combine the two dfs:
    slips = pd.merge(slip_seqs, slip_press, on='sub',
                     how='outer', suffixes=('_seq', '_press'))
    # now combine thisk with the long_df:
    long_df = pd.merge(long_df, slips, on='sub', how='outer')
    # now where nan in the slips, put 0:
    long_df = long_df.fillna(0)
    calculate_accuracy(gambles, long_df, utility_power)
    # calculate_accuracy(main_data_df, long_df, utility_power)

    # change accuracy to riskModel_accuracy:
    long_df_temp = long_df.copy()
    long_df_temp = long_df_temp.rename(columns={'accuracy': 'riskModel_accuracy'})
    long_df_temp = long_df_temp.drop(columns=['block'])
    long_df_temp = long_df_temp.drop(columns=['sequenceCompleted'])
    long_df_temp = long_df_temp.drop(columns=['SRO_rt_of_SRO_key'])
    long_df_temp
    # merge the two dfs (but do it on sub group and block):
    main_data_df = pd.merge(main_data_df, long_df_temp, on=[
                            'sub', 'group'], how='outer')
    main_data_df
    # now same for gambles:
    gambles = pd.merge(gambles, long_df_temp, on=['sub', 'group'], how='outer')
    gambles


# now calculate the subjective utility for each gamble:
# first calculate the utility for each gamble:
gambles['gambles_bottom_subj_utility'] = gambles.apply(
    lambda x: x['gambles_bottom'][1] * utility_power(x['gambles_bottom'][0], x['rho']), axis=1)
gambles['gambles_top_subj_utility'] = gambles.apply(
    lambda x: x['gambles_top'][1] * utility_power(x['gambles_top'][0], x['rho']), axis=1)

gambles['utility_diff'] = np.abs(
    gambles['gambles_bottom_subj_utility'] - gambles['gambles_top_subj_utility'])
gambles['utility_diff_abs_log_ratio'] = np.abs(np.log(
    gambles['gambles_bottom_subj_utility'] / gambles['gambles_top_subj_utility']))


gambles['utility_diff_bin'] = pd.qcut(
    gambles['utility_diff'], 2, labels=['hard', 'easy'])
gambles['utility_diff_bin'] = gambles['utility_diff_bin'].astype('object')
gambles['utility_diff_abs_log_ratio_bin'] = pd.qcut(
    gambles['utility_diff_abs_log_ratio'], 2, labels=['hard', 'easy'])
gambles['utility_diff_abs_log_ratio_bin'] = gambles['utility_diff_abs_log_ratio_bin'].astype(
    'object')

# ---------------------------------------------------------

# function to calculate accuracies:


def add_choice_prediction(gamble_data):

    # iterate rows in fitted_data and calculate accuracy for each subject
    for i, row in gamble_data.iterrows():
        # check if row.chosen_gamble is a list:
        if isinstance(row.chosen_gamble, list):
            # subject/line model data:
            b = row.b

            # Get the choices, probabilities, and payoffs
            choice = 1 if row['chose_bottom'] == True else 0
            EU1 = row['gambles_bottom_subj_utility']
            EU2 = row['gambles_top_subj_utility']

            # Calculate the model's predicted choices
            prob_choice1 = 1 / (1 + np.exp(-b*(EU1-EU2)))
            predicted_choose_bottom = 1 if prob_choice1 >= 0.5 else 0

            prediction_success = 1 if predicted_choose_bottom == choice else 0

            # add value in a new column predicted_choose_bottom:
            gamble_data.loc[i,
                            'predicted_choose_bottom'] = predicted_choose_bottom
            gamble_data.loc[i, 'prediction_success'] = prediction_success
        else:
            gamble_data.loc[i, 'predicted_choose_bottom'] = np.nan
            gamble_data.loc[i, 'prediction_success'] = np.nan


add_choice_prediction(gambles)

# now I want the columsn from gambles: utility_diff utility_diff_abs_log_ratio utility_diff_bin utility_diff_abs_log_ratio_bin and prediction_success in main_data_df
# now merge the two dfs:
main_data_df = pd.merge(main_data_df, gambles[['sub', 'group', 'block', 'trial', 'utility_diff', 'utility_diff_abs_log_ratio',
                        'utility_diff_bin', 'utility_diff_abs_log_ratio_bin', 'prediction_success']], on=['sub', 'group', 'block', 'trial'], how='outer')
main_data_df

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# %%
