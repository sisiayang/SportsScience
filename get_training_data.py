import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import random as rd

def permute_sequence(df):
    rally_list = []
    new_df = pd.DataFrame(columns=df.columns)
    for _, rally in df.groupby(['Game', 'Rally']):
        curr_team = df.iloc[rally.index[0]]['Team']
        shuffle_list = []
        tmp_list = []
        for _, row in rally.iterrows():
            if(row['Team'] != curr_team):
                shuffle_list.append(tmp_list)
                tmp_list = []
                curr_team = row['Team']
            tmp_list.append(row.tolist())
        shuffle_list.append(tmp_list)
        ############## shuffle ########################
        shuffled_list = shuffle(shuffle_list, random_state=3)
        restore_list = []
        for i in shuffled_list:
            restore_list += i
        shuffled_df = pd.DataFrame(restore_list, columns=df.columns)
        ############## combine df #####################]
        rally_list.append(shuffled_df)

    rd.shuffle(rally_list)
    new_df = pd.DataFrame(columns=df.columns)
    for rally in rally_list:
        new_df = pd.concat([new_df, rally])
    return new_df

def permute_feature(df, feature):
    if(feature == 'Original'):
        return df
    elif(feature == 'Sequence'):
        return permute_sequence(df)
    shuffled_feature = shuffle(df[feature], random_state=3)
    shuffled_feature = shuffled_feature.reset_index(drop=True)
    df[feature] = shuffled_feature
    return df

def get_rally_result(df):
    result_label = []
    ignor_game_rally_index = []
    team_col = [c for c in df.columns if 'Team' in c]
    for _, df_rally in df.groupby(['Game', 'Rally']):
        start_from = df.iloc[df_rally.index[0]][team_col].tolist()

        result = df_rally[df_rally['Score'] == 1]
        if(len(result) != 0):
            if(result.iloc[0][team_col].tolist() == start_from):
                result_label.append([0, 1])
            else:
                result_label.append([1, 0])
        else:
            result = df_rally[df_rally['Errors'] == 1]
            if(len(result) == 0):
                ignor_game_rally_index.append((df_rally.iloc[0]['Game'], df_rally.iloc[0]['Rally']))
            elif(result.iloc[0][team_col].tolist() != start_from):
                result_label.append([0, 1])
            else:
                result_label.append([1, 0])
    return tf.constant(result_label, dtype=float), ignor_game_rally_index



def get_padding_data(df, ignor):
    space_col = [c for c in df.columns if 'Space' in c]
    action_col = [c for c in df.columns if 'Action' in c]
    result_col = ['Errors', 'Score', 'Nothing']
    others_col = [c for c in df.columns if c not in space_col and c not in action_col and c not in result_col and c != 'Game' and c != 'Rally' and c != 'Team']
    team_col = [c for c in df.columns if 'Team' in c]

    rally_set = []
    rally_space_set = []
    rally_action_set = []
    for _, df_rally in df.groupby(['Game', 'Rally']):   # each rally in one game
        if((df_rally.iloc[0]['Game'], df_rally.iloc[0]['Rally']) in ignor):
            continue
        start_from = df.iloc[df_rally.index[0]][team_col].tolist()
        shot_set = []
        shot_space_set = []
        shot_action_set = []
        
        atk_sequence = []
        atk_space_sequence = []
        atk_action_sequence = []

        curr_team = start_from
        for _, shot in df_rally.iterrows():
            if(shot[team_col].tolist() != curr_team or len(atk_sequence) == 3):
                shot_set.append(atk_sequence)
                shot_space_set.append(atk_space_sequence)
                shot_action_set.append(atk_action_sequence)
                
                curr_team = shot[team_col].tolist()

                atk_sequence = []
                atk_space_sequence = []
                atk_action_sequence = []

            atk_space_sequence.append(shot[space_col])
            atk_action_sequence.append(shot[action_col])
            atk_sequence.append(shot[others_col])
        
        # the last shot
        shot_set.append(atk_sequence)
        shot_space_set.append(atk_space_sequence)
        shot_action_set.append(atk_action_sequence)

        # one rally has been finished
        shot_set = pad_sequences(shot_set, maxlen=3, padding='post')
        shot_space_set = pad_sequences(shot_space_set, maxlen=3, padding='post')
        shot_action_set = pad_sequences(shot_action_set, maxlen=3, padding='post')

        # one rally has been finished
        rally_set.append(shot_set)
        rally_space_set.append(shot_space_set)
        rally_action_set.append(shot_action_set)

    padded_rally_set = pad_sequences(rally_set, dtype=float, maxlen=19, padding='post')
    padded_rally_space_set = pad_sequences(rally_space_set, dtype=float, maxlen=19, padding='post')
    padded_rally_action_set = pad_sequences(rally_action_set, dtype=float, maxlen=19, padding='post')
    
    return padded_rally_set, padded_rally_space_set, padded_rally_action_set