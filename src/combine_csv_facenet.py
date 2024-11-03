import pandas as pd
import numpy as np

pids = open('../data/processed/p_ids.txt').read().splitlines()

fn_dfs = []
for pid in pids:
    # pid='P006'

    # Get np arrays
    arr_I = np.load(f'../data/processed/deepface/facenet/{pid}_I_embeddings.npz')
    arr_P = np.load(f'../data/processed/deepface/facenet/{pid}_P_embeddings.npz')
    
    # Convert np to dataframes
    df_I = pd.DataFrame(arr_I['embeddings'])
    df_I.columns = [f'e{c}' for c in df_I.columns]
    df_I['face_confidence'] = arr_I['face_confidences']
    df_I.columns = [f'fn_{c}_I' for c in df_I.columns]
    
    df_P = pd.DataFrame(arr_P['embeddings'])
    df_P.columns = [f'e{c}' for c in df_P.columns]
    df_P['face_confidence'] = arr_P['face_confidences']
    df_P.columns = [f'fn_{c}_P' for c in df_P.columns]
    df_P['frame'] = arr_P['frame_numbers']

    df_fn = pd.concat((df_I, df_P), axis=1).reset_index(drop=True)

    df_fn['PID'] = pid

    fn_dfs.append(df_fn)
    
master_fn_df = pd.concat(fn_dfs, axis=0).reset_index(drop=True)

master_fn_df.to_csv('../data/processed/deepface/facenet/facenet_master.csv', index=False)

# a = np.load(f'../data/processed/deepface/facenet/P007_P_embeddings.npz')
# a['embeddings'].shape[0]
# a['frame_numbers']
# a['face_confidences']

# tot_a = np.hstack((a['frame_numbers'].reshape([-1,1]), a['face_confidences'].reshape([-1,1]), a['embeddings']))
    
# d = pd.DataFrame(a['embeddings'])
# d.info()
# d.head()
# d.columns = [f'e{c}' for c in d.columns]
# d.head()
# d['frame'] = a['frame_numbers']
# d['face_confidence'] = a['face_confidences']

