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

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import math
from tqdm import tqdm

#######
####### Using master data csv file
#######

df_master = pd.read_csv('../data/processed/master_features.csv')
df_Y = pd.read_csv('../data/raw/Questionnaires/Interviewer_EvaluationScore.csv')


### Get average of:
# Features to investigate:
# 1. Eye Gaze direction
# 16 * 5 = 80
f1s = ["of_gaze_0_x_I", "of_gaze_0_y_I", "of_gaze_0_z_I", "of_gaze_1_x_I", "of_gaze_1_y_I", "of_gaze_1_z_I", "of_gaze_angle_x_I", "of_gaze_angle_y_I",
       "of_gaze_0_x_P", "of_gaze_0_y_P", "of_gaze_0_z_P", "of_gaze_1_x_P", "of_gaze_1_y_P", "of_gaze_1_z_P", "of_gaze_angle_x_P", "of_gaze_angle_y_P"]
# 2. Face Pose location and rotation
# 12 * 5 = 60
f2s = ["of_pose_Tx_I", "of_pose_Ty_I", "of_pose_Tz_I", "of_pose_Rx_I", "of_pose_Ry_I", "of_pose_Rz_I",
       "of_pose_Tx_P", "of_pose_Ty_P", "of_pose_Tz_P", "of_pose_Rx_P", "of_pose_Ry_P", "of_pose_Rz_P"]
# 3. Intensity of 17 facial AU's
# 34 * 5 = 170
f3s =  ["of_AU01_r_I", "of_AU02_r_I", "of_AU04_r_I", "of_AU05_r_I", "of_AU06_r_I", "of_AU07_r_I", "of_AU09_r_I", "of_AU10_r_I", "of_AU12_r_I", "of_AU14_r_I", "of_AU15_r_I", "of_AU17_r_I", "of_AU20_r_I", "of_AU23_r_I", "of_AU25_r_I", "of_AU26_r_I", "of_AU45_r_I",
       "of_AU01_r_P", "of_AU02_r_P", "of_AU04_r_P", "of_AU05_r_P", "of_AU06_r_P", "of_AU07_r_P", "of_AU09_r_P", "of_AU10_r_P", "of_AU12_r_P", "of_AU14_r_P", "of_AU15_r_P", "of_AU17_r_P", "of_AU20_r_P", "of_AU23_r_P", "of_AU25_r_P", "of_AU26_r_P", "of_AU45_r_P"]

Y_cols = ['PID', 'Hirability_Interview', 'Evaluation Score_POST', 'Confident', 'Stressed', 'Uncomfortable']

# Get sub-dataframe with selected features and 'PID'
df_X = df_master[f1s + f2s + f3s + ['PID']]

# df_X_avg = df_X.groupby('PID').mean().reset_index()

# Compute mean, variance, max, min, and standard deviation per PID
df_X_agg = df_X.groupby('PID').agg(['mean', 'var', 'max', 'min', 'std']).reset_index()

# Flatten the columns
df_X_agg.columns = ['PID'] + ['{}_{}'.format(col[0], col[1]) for col in df_X_agg.columns[1:]]

# Define the statistical measures
# TODO: choose var or std
stats = ['mean', 'var', 'max', 'min', 'std']

# Create new feature lists with statistics
new_features = ['{}_{}'.format(feat, stat) for feat in f1s + f2s + f3s for stat in stats]

# Similarly, create feature lists for each feature group
f1s_stats = ['{}_{}'.format(feat, stat) for feat in f1s for stat in stats]
f2s_stats = ['{}_{}'.format(feat, stat) for feat in f2s for stat in stats]
f3s_stats = ['{}_{}'.format(feat, stat) for feat in f3s for stat in stats]

# Get Y's to predict

df_Y = df_Y[Y_cols]

# Filter out PID's that aren't in the master df (for example the videos that do switching and we couldn't process)
df_Y = df_Y[df_Y['PID'].isin(df_X_agg['PID'])].reset_index(drop=True)

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

df_Y_bin = df_Y.copy()

df_Y_bin['Hirability_Interview'] = df_Y_bin['Hirability_Interview'].apply(lambda x: 1 if x == 5 else 0)
df_Y_bin['Evaluation Score_POST'] = df_Y_bin['Evaluation Score_POST'].apply(lambda x: 1 if x >= 4.5 else 0)
df_Y_bin['Confident'] = df_Y_bin['Confident'].apply(lambda x: 1 if x == 5 else 0)
df_Y_bin['Stressed'] = df_Y_bin['Stressed'].apply(lambda x: 1 if x >= 3 else 0)
df_Y_bin['Uncomfortable'] = df_Y_bin['Uncomfortable'].apply(lambda x: 1 if x >= 2 else 0)

#####
##### Normalize All The Features Including the target
#####

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

scaler_X.fit(df_X_agg[new_features])
scaler_Y.fit(df_Y[Y_cols[1:]])

df_X_scaled = df_X_agg.copy()
df_X_scaled[new_features] = scaler_X.transform(df_X_scaled[new_features])

df_Y_scaled = df_Y.copy()
df_Y_scaled[Y_cols[1:]] = scaler_Y.transform(df_Y_scaled[Y_cols[1:]])

df_X_scaled.drop(index=1)
for i in df_X_scaled.index:
    print(i)
    
def generate_hyperparameters(base_params, num_features):
    dynamic_params = {}
    
    print(base_params)
    
    for key, value in base_params.items():
        if callable(value):
            dynamic_params[key] = value(num_features)
        else:
            dynamic_params[key] = value

    return dynamic_params


def eval_features(df_X, df_Y, X_features, Y_feature, model_class, base_hyper_params, is_regression=True):
    
    num_features = len(X_features)
    
    hyper_parameters = generate_hyperparameters(base_hyper_params, num_features)
    
    param_grid = ParameterGrid(hyper_parameters)
    
    predictions = []
    actuals = []
    
    for cv_outer_i in tqdm(df_X.index):
        cv_outer_x = df_X.iloc[[cv_outer_i]]
        cv_outer_y = df_Y.iloc[cv_outer_i][Y_feature]
        
        train_outer_x = df_X.drop(index=cv_outer_i).reset_index(drop=True)
        train_outer_y = df_Y.drop(index=cv_outer_i)[Y_feature].reset_index(drop=True)
        
        best_hp = None
        if is_regression:
            best_hp_avg_metric = float('inf') # because error
        else:
            best_hp_avg_metric = float('-inf') # becuase accuracy
        
        for params in param_grid:
            
            metrics = []
        
            for cv_inner_i in train_outer_x.index:
                cv_inner_x = train_outer_x.iloc[[cv_inner_i]]
                cv_inner_y = train_outer_y.iloc[cv_inner_i]
                
                train_inner_x = train_outer_x.drop(index=cv_inner_i)
                train_inner_y = train_outer_y.drop(index=cv_inner_i)
                
                
                model = model_class(**params, random_state=1)
                
                model.fit(train_inner_x, train_inner_y)
                
                pred = model.predict(cv_inner_x)
                actual = cv_inner_y
                if is_regression:
                    # Absolute Error
                    metric = abs(pred - actual)
                else:
                    # Accuracy
                    metric = int(pred == actual)
                
                metrics.append(metric)
            
            avg_metric = np.array(metrics).mean()
            
            if ((is_regression and best_hp_avg_metric > avg_metric) # bc Mean Absolute Error
                or 
                (not is_regression and best_hp_avg_metric < avg_metric)): # bc Accuracy
                best_hp = params
                best_hp_avg_metric = avg_metric
                
        # print(f'best depth at cv {cv_outer_i}: {best_hp} - score: {best_hp_avg_score}')
        
        model = model_class(**best_hp , random_state=1)
        
        model.fit(train_outer_x, train_outer_y)
        
        predictions.append(model.predict(cv_outer_x))
        actuals.append(cv_outer_y)
    
    if is_regression:
        final_mae = abs((np.array(predictions) - np.array(actuals))).mean()
        print(f'Regression for {Y_feature}, final MAE: {final_mae:.3f}')
    else:
        final_acc = (np.array(predictions) == np.array(actuals)).mean()
        print(f'Classification for {Y_feature:<20}, final MAE: {final_mae:.3f}')


# Decision Tree
dt_hyper_params = {
    'max_depth': lambda n: [int(0.5*math.sqrt(n)), int(1*math.sqrt(n)), int(1.5*math.sqrt(n)), int(2*math.sqrt(n))]
}

# XGBoost Hyper Parameters
# TODO: HP Tunning: depth (0.5*sqrt(#features), sqrt(#features), 1.5*sqrt(#features), 2*sqrt(#features))
        #                   # of trees (10,20,30,40)
xgb_hyper_params = {
    'max_depth': lambda n: [int(0.5*math.sqrt(n)), int(1*math.sqrt(n)), int(1.5*math.sqrt(n)), int(2*math.sqrt(n))],
    'n_estimators': [20,30,50,100]
}

# FFNN Hyper Parameters
# TODO: HP Tunning: # of nodes in layer (#features/3, #features/2, #features), 
# drop-out (0.05, 0.10, 0.15, 0.20), 
# regularization (0.0001, 0.0005, 0.001)
# def FFNN(l1_size, l2_size, drop_out, regularization):
    


eval_features(df_X_scaled[f1s_stats], df_Y_scaled, f1s_stats, 'Uncomfortable', tree.DecisionTreeRegressor, dt_hyper_params, True)       





df_X_scaled[f1s_stats].drop(index=0).iloc[1].values.reshape(1,-1)        

for feature_set in [f1s_stats, f2s_stats, f3s_stats]:
    for y_feat in Y_cols[1:]:
        
        eval_features(df_X_scaled[feature_set], df_Y_scaled, feature_set, y_feat)











### Split Train and Test

X_train, X_test, y_train, y_test = train_test_split(df_X_agg, df_Y, test_size=0.2, random_state=1)

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

scaler_X.fit(X_train[new_features])
scaler_Y.fit(y_train[Y_cols[1:]])

# Scale Train

X_train_scaled = X_train.copy()
X_train_scaled[new_features] = scaler_X.transform(X_train[new_features])

y_train_scaled = y_train.copy()
y_train_scaled[Y_cols[1:]] = scaler_Y.transform(y_train[Y_cols[1:]])

# Scale Test

X_test_scaled = X_test.copy()
X_test_scaled[new_features] = scaler_X.transform(X_test[new_features])

y_test_scaled = y_test.copy()
y_test_scaled[Y_cols[1:]] = scaler_Y.transform(y_test[Y_cols[1:]])

#####
##### Test All Features
#####

def train_and_eval_model(X_train, X_test, y_train, y_test, model, scorer):
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=LeaveOneOut(), scoring=scorer)
    
    # Train on total train set and eval on test set
    model.fit(X_train, y_train)
    
    train_score = scorer(model, X_train, y_train)
    test_score = scorer(model, X_test, y_test)
    
    print(f"{model.__class__.__name__:<25} -- Train score: {train_score:.3f} -- Average CV score: {cv_scores.mean():.3f} -- Test Score: {test_score:.3f}")

##### Regression

# For each feature set and then for each predictor/output
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# TODO: mean absolute error, pearsons correlation
# TODO: Pearsons: save pred over ea

for f in [f1s_stats, f2s_stats, f3s_stats]:
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

for f in [f1s_stats, f2s_stats, f3s_stats]:
    print(f"Testing features: {f}")
    X_train_f = X_train_scaled[f]
    X_test_f = X_test_scaled[f]
    
    for i in range(1,len(Y_cols)):
        print(f'\n--- {Y_cols[i]} ---')
        y_train_f = y_train_bin[Y_cols[i]]
        y_test_f = y_test_bin[Y_cols[i]]
        
        # Models: decision tree, xgboost, mlp
        # TODO: HP Tunning: depth (0.5*sqrt(#features), sqrt(#features), 1.5*sqrt(#features), 2*sqrt(#features))
        dt = tree.DecisionTreeClassifier(random_state=1)
        # TODO: HP Tunning: depth (0.5*sqrt(#features), sqrt(#features), 1.5*sqrt(#features), 2*sqrt(#features))
        #                   # of trees (10,20,30,40)
        xgbc = xgb.XGBClassifier(random_state=1)
        # TODO: HP Tunning: # of nodes in layer (#features/3, #features/2, #features), drop-out (0.05, 0.10, 0.15, 0.20), regularization (0.0001, 0.0005, 0.001)
        mlp = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam',
                        random_state=1, max_iter=1000)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, dt, acc_scorer)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, xgbc, acc_scorer)
        train_and_eval_model(X_train_f, X_test_f, y_train_f, y_test_f, mlp, acc_scorer)
        

def eval_features(df_X, df_Y, X_features, Y_features):
    
    num_features = len(X_features)
    # Decision Tree
    hyper_parameters = {
        'max_depth': [0.5*math.sqrt(num_features), 1*math.sqrt(num_features), 1.5*math.sqrt(num_features), 2*math.sqrt(num_features)]
    }
    
    for cv_outer_i in range(0,len(df_X)):
        cv_outer_x = df_X.iloc[cv_outer_i]
        cv_outer_y = df_X.iloc[cv_outer_i]
        
    
