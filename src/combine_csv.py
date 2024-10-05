### Combine all of the feature csv files (except the deepface embeddings) into one master csv

import pandas as pd
import numpy as np

pids = open('../data/processed/p_ids.txt').read().splitlines()

### RLOF

rlof_dfs = []
for pid in pids:

    df_I = pd.read_csv(f'../data/processed/rlof/{pid}_I_rlof.csv')
    df_P = pd.read_csv(f'../data/processed/rlof/{pid}_P_rlof.csv')

    # 2 frame columns is redundant
    df_P = df_P.drop('frame', axis = 1)

    df_I.columns = 'rlof_' + df_I.columns.str.strip() + '_I'
    df_P.columns = 'rlof_' + df_P.columns.str.strip() + '_P'

    df_rlof = pd.concat((df_I, df_P), axis=1)

    df_rlof['PID'] = pid

    rlof_dfs.append(df_rlof)
    
master_rlof_df = pd.concat(rlof_dfs, axis=0).reset_index(drop=True)
# master_rlof_df.info()
# master_rlof_df.head()

### WMEI

## IMPORTANT: the frame indexes are off by 1, they should be 1 less than they are in the files.

wmei_dfs = []
for pid in pids:

    df_I = pd.read_csv(f'../data/processed/wmei/{pid}_I_wmei_stats.csv')
    df_P = pd.read_csv(f'../data/processed/wmei/{pid}_P_wmei_stats.csv')

    # 2 frame columns is redundant
    df_P = df_P.drop('frame', axis = 1)
    
    # FIX FRAME ERROR
    df_I['frame'] = df_I['frame'] - 1

    df_I.columns = 'wmei_' + df_I.columns.str.strip() + '_I'
    df_P.columns = 'wmei_' + df_P.columns.str.strip() + '_P'

    df_wmei = pd.concat((df_I, df_P), axis=1)

    df_wmei['PID'] = pid

    wmei_dfs.append(df_wmei)
    
master_wmei_df = pd.concat(wmei_dfs, axis=0).reset_index(drop=True)
# master_wmei_df.info()
# master_wmei_df.head()

### OpenFace

of_dfs = []
for pid in pids:
    # pid='P006'

    df_I = pd.read_csv(f'../data/processed/openface/{pid}_I.csv')
    df_P = pd.read_csv(f'../data/processed/openface/{pid}_P.csv')

    # Drop 2d coordinate values like 'eye_lmk_x_{0-55}' 'eye_lmk_y_{0-55}' 'x_{0-65}' 'y_{0-65}'
    df_I = df_I.drop(columns=df_I.filter(regex=r'(eye_lmk_[xy]_\d+|[xy]_\d+)'))
    df_P = df_P.drop(columns=df_P.filter(regex=r'(eye_lmk_[xy]_\d+|[xy]_\d+)'))

    # 2 frame columns is redundant
    df_P = df_P.drop('frame', axis = 1)

    df_I.columns = 'of_' + df_I.columns.str.strip() + '_I'
    df_P.columns = 'of_' + df_P.columns.str.strip() + '_P'

    df_of = pd.concat((df_I, df_P), axis=1)

    df_of['PID'] = pid

    of_dfs.append(df_of)
    
master_of_df = pd.concat(of_dfs, axis=0).reset_index(drop=True)
# master_of_df.info()
# master_of_df.head()

### Verify Frames are correct

# for i in range(len(wmei_dfs)):
#     df_1 = wmei_dfs[i]['wmei_frame_I'].max() - wmei_dfs[i]['wmei_frame_I'].min()
#     df_2 = rlof_dfs[i]['rlof_frame_I'].max() - rlof_dfs[i]['rlof_frame_I'].min()
#     df_3 = of_dfs[i]['of_frame_I'].max() - of_dfs[i]['of_frame_I'].min()
    
#     print(df_1, df_2, df_3)

### Combine into single dataframe and export to csv

print('starting 1st merge')

temp_df = pd.merge(master_rlof_df, master_wmei_df, left_on=['rlof_frame_I', 'PID'], right_on=['wmei_frame_I', 'PID'], how='inner')

print('starting main merge')

master_df = pd.merge(temp_df, master_of_df, left_on=['rlof_frame_I', 'PID'], right_on=['of_frame_I', 'PID'], how='inner')

print('starting export to csv')

master_df.to_csv('../data/processed/master_features.csv')

    
