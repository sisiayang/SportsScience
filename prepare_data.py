import pandas as pd
import sys

df_IRI = pd.read_excel(sys.argv[0], sheet_name=sys.argv[1])
df_IRI = df_IRI.drop(columns=['Unnamed: 6'])

# 處理 data
df_IRI = df_IRI.fillna(0)
df_IRI = df_IRI.drop(len(df_IRI)-1)
df_IRI.insert(0, column='Team', value=['IRI' for i in range(len(df_IRI))])
df_IRI = df_IRI.drop(df_IRI[df_IRI['Action'] == 'OP'].index)
df_IRI = df_IRI.reset_index(drop=True)

# get_dummy
df_IRI = df_IRI.astype({'No.': 'int32'})
df_IRI = df_IRI.astype({'No.': 'str'})
df_IRI_getdum = pd.get_dummies(df_IRI)

# split to process
space_col = [c for c in df_IRI_getdum.columns if c.startswith('Space')]
action_col = [c for c in df_IRI_getdum.columns if c.startswith('Action')]
other_col = [c for c in df_IRI_getdum.columns if(c not in space_col and c not in action_col)]
space_encoding = df_IRI_getdum[space_col].values.astype('float32')
action_encoding = df_IRI_getdum[action_col].values.astype('float32')
other_encoding = df_IRI_getdum[other_col].values.astype('float32')