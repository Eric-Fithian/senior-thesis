# Features to investigate:
# 1. Eye Gaze direction
# 2. Face Pose location and rotation
# 3. Intensity of 17 facial AU's

# Outputs:
# 1. Hireability_Interview
# 2. Evaluation_score-POST
# 3. Confident
# 4. Stressed
# 5. Uncomfortable

# Classification Types:
# 1. Regression
# 2. Binary

# Models:
# 1. Decision Tree
# 2. XGBoost
# 3. FFNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

#######
####### Using master data csv file
#######

df_master = pd.read_csv('../data/processed/master_features.csv')
df_Y = pd.read_csv('../data/raw/Questionnaires/Interviewer_EvaluationScore.csv')


### Get average of:
# Features to investigate:
# 1. Eye Gaze direction
f1s = ["of_gaze_0_x_I", "of_gaze_0_y_I", "of_gaze_0_z_I", "of_gaze_1_x_I", "of_gaze_1_y_I", "of_gaze_1_z_I", "of_gaze_angle_x_I", "of_gaze_angle_y_I",
       "of_gaze_0_x_P", "of_gaze_0_y_P", "of_gaze_0_z_P", "of_gaze_1_x_P", "of_gaze_1_y_P", "of_gaze_1_z_P", "of_gaze_angle_x_P", "of_gaze_angle_y_P"]
# 2. Face Pose location and rotation
f2s = ["of_pose_Tx_I", "of_pose_Ty_I", "of_pose_Tz_I", "of_pose_Rx_I", "of_pose_Ry_I", "of_pose_Rz_I",
       "of_pose_Tx_P", "of_pose_Ty_P", "of_pose_Tz_P", "of_pose_Rx_P", "of_pose_Ry_P", "of_pose_Rz_P"]
# 3. Intensity of 17 facial AU's
f3s =  ["of_AU01_r_I", "of_AU02_r_I", "of_AU04_r_I", "of_AU05_r_I", "of_AU06_r_I", "of_AU07_r_I", "of_AU09_r_I", "of_AU10_r_I", "of_AU12_r_I", "of_AU14_r_I", "of_AU15_r_I", "of_AU17_r_I", "of_AU20_r_I", "of_AU23_r_I", "of_AU25_r_I", "of_AU26_r_I", "of_AU45_r_I",
       "of_AU01_r_P", "of_AU02_r_P", "of_AU04_r_P", "of_AU05_r_P", "of_AU06_r_P", "of_AU07_r_P", "of_AU09_r_P", "of_AU10_r_P", "of_AU12_r_P", "of_AU14_r_P", "of_AU15_r_P", "of_AU17_r_P", "of_AU20_r_P", "of_AU23_r_P", "of_AU25_r_P", "of_AU26_r_P", "of_AU45_r_P"]

Y_cols = ['PID', 'Hirability_Interview', 'Evaluation Score_POST', 'Confident', 'Stressed', 'Uncomfortable']


# Get sub-dataframe with selected features and 'PID'
df_X = df_master[f1s + f2s + f3s + ['PID']]

df_X_avg = df_X.groupby('PID').mean().reset_index()

# Get Y's to predict

df_Y = df_Y[Y_cols]

# Filter out PID's that aren't in the master df (for example the videos that do switching and we couldn't process)
df_Y = df_Y[df_Y['PID'].isin(df_X_avg['PID'])]

### Split Train and Test

X_train, X_test, y_train, y_test = train_test_split(df_X_avg, df_Y, test_size=0.2, random_state=1)

#####
##### Binarize data for binary classification later
##### 

## Visualize Target/Output Variables

df_Y[1:].hist(figsize=(6,8))

# Based on Histogram's
# Hirability_Interview:  [0, 4],    [5]
# Evaluation Score_POST: [0, 4.5),  [4.5, 5] (not discrete)
# Confident:             [0, 4],    [5]
# Stressed:              [0, 2],    [3,5]
# Uncomfortable:         [1],       [2,3]

y_train_bin, y_test_bin = y_train.copy(), y_test.copy()

y_train_bin['Hirability_Interview'] = y_train_bin['Hirability_Interview'].apply(lambda x: 1 if x == 5 else 0)
y_train_bin['Evaluation Score_POST'] = y_train_bin['Evaluation Score_POST'].apply(lambda x: 1 if x >= 4.5 else 0)
y_train_bin['Confident'] = y_train_bin['Confident'].apply(lambda x: 1 if x == 5 else 0)
y_train_bin['Stressed'] = y_train_bin['Stressed'].apply(lambda x: 1 if x >= 3 else 0)
y_train_bin['Uncomfortable'] = y_train_bin['Uncomfortable'].apply(lambda x: 1 if x >= 2 else 0)

y_test_bin['Hirability_Interview'] = y_test_bin['Hirability_Interview'].apply(lambda x: 1 if x == 5 else 0)
y_test_bin['Evaluation Score_POST'] = y_test_bin['Evaluation Score_POST'].apply(lambda x: 1 if x >= 4.5 else 0)
y_test_bin['Confident'] = y_test_bin['Confident'].apply(lambda x: 1 if x == 5 else 0)
y_test_bin['Stressed'] = y_test_bin['Stressed'].apply(lambda x: 1 if x >= 3 else 0)
y_test_bin['Uncomfortable'] = y_test_bin['Uncomfortable'].apply(lambda x: 1 if x >= 2 else 0)

#####
##### Normalize All The Features Including the target
#####

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

scaler_X.fit(X_train[f1s+f2s+f3s])
scaler_Y.fit(y_train[Y_cols[1:]])

# Scale Train

X_train_scaled = X_train.copy()
X_train_scaled[f1s+f2s+f3s] = scaler_X.transform(X_train[f1s+f2s+f3s])

y_train_scaled = y_train.copy()
y_train_scaled[Y_cols[1:]] = scaler_Y.transform(y_train[Y_cols[1:]])

# Scale Test

X_test_scaled = X_test.copy()
X_test_scaled[f1s+f2s+f3s] = scaler_X.transform(X_test[f1s+f2s+f3s])

y_test_scaled = y_test.copy()
y_test_scaled[Y_cols[1:]] = scaler_Y.transform(y_test[Y_cols[1:]])

#####
##### Test All Features
#####

def train_and_eval_model(X_train, X_test, y_train, y_test, model, scorer):
    
    # Cross validation
    scores = cross_val_score(model, X_train, y_train, cv=LeaveOneOut(), scoring=scorer)
    
    # Train on total train set and eval on test set
    model.fit(X_train, y_train)
    
    score = scorer(model, X_test, y_test)
    
    print(f"{model.__class__.__name__:<25} -- Average CV score: {scores.mean():.3f} -- Test Score: {score:.3f}")

##### Regression

# For each feature set and then for each predictor/output
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

for f in [f1s, f2s, f3s]:
    print(f"Testing features: {f}")
    X_train_f = X_train_scaled[f]
    X_test_f = X_test_scaled[f]
    
    for i in range(1,len(Y_cols)):
        print(f'\n--- {Y_cols[i]} ---')
        y_train_f = y_train_scaled[Y_cols[i]]
        y_test_f = y_test_scaled[Y_cols[i]]
        
        # Models: decision tree, xgboost, mlp
        dt = tree.DecisionTreeRegressor(random_state=1)
        xgbr = xgb.XGBRegressor(random_state=1)
        mlp = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam',
                        random_state=1, max_iter=1000)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, dt, mse_scorer)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, xgbr, mse_scorer)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, mlp, mse_scorer)


##### Binary Classification

# For each feature set and then for each predictor/output
acc_scorer = make_scorer(accuracy_score)

for f in [f1s, f2s, f3s]:
    print(f"Testing features: {f}")
    X_train_f = X_train_scaled[f]
    X_test_f = X_test_scaled[f]
    
    for i in range(1,len(Y_cols)):
        print(f'\n--- {Y_cols[i]} ---')
        y_train_f = y_train_bin[Y_cols[i]]
        y_test_f = y_test_bin[Y_cols[i]]
        
        # Models: decision tree, xgboost, mlp
        dt = tree.DecisionTreeClassifier(random_state=1)
        xgbc = xgb.XGBClassifier(random_state=1)
        mlp = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam',
                        random_state=1, max_iter=1000)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, dt, acc_scorer)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, xgbc, acc_scorer)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, mlp, acc_scorer)
