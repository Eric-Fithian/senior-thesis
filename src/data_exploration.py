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
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn import tree
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error, r2_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import (GridSearchCV, KFold, LeaveOneOut,
                                     ParameterGrid, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from tensorflow.keras import layers, models, regularizers
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
f3s = ["of_AU01_r_I", "of_AU02_r_I", "of_AU04_r_I", "of_AU05_r_I", "of_AU06_r_I", "of_AU07_r_I", "of_AU09_r_I", "of_AU10_r_I", "of_AU12_r_I", "of_AU14_r_I", "of_AU15_r_I", "of_AU17_r_I", "of_AU20_r_I", "of_AU23_r_I", "of_AU25_r_I", "of_AU26_r_I", "of_AU45_r_I",
       "of_AU01_r_P", "of_AU02_r_P", "of_AU04_r_P", "of_AU05_r_P", "of_AU06_r_P", "of_AU07_r_P", "of_AU09_r_P", "of_AU10_r_P", "of_AU12_r_P", "of_AU14_r_P", "of_AU15_r_P", "of_AU17_r_P", "of_AU20_r_P", "of_AU23_r_P", "of_AU25_r_P", "of_AU26_r_P", "of_AU45_r_P"]
# 4. Point Distribution Model
f4s = ["of_p_scale_I", "of_p_rx_I", "of_p_ry_I", "of_p_rz_I", "of_p_tx_I", "of_p_ty_I", "of_p_0_I", "of_p_1_I", "of_p_2_I", "of_p_3_I", "of_p_4_I", "of_p_5_I", "of_p_6_I", "of_p_7_I", "of_p_8_I", "of_p_9_I", "of_p_10_I", "of_p_11_I", "of_p_12_I", "of_p_13_I", "of_p_14_I", "of_p_15_I", "of_p_16_I", "of_p_17_I", "of_p_18_I", "of_p_19_I", "of_p_20_I", "of_p_21_I", "of_p_22_I", "of_p_23_I", "of_p_24_I", "of_p_25_I", "of_p_26_I", "of_p_27_I", "of_p_28_I", "of_p_29_I", "of_p_30_I", "of_p_31_I", "of_p_32_I", "of_p_33_I", 
       "of_p_scale_P", "of_p_rx_P", "of_p_ry_P", "of_p_rz_P", "of_p_tx_P", "of_p_ty_P", "of_p_0_P", "of_p_1_P", "of_p_2_P", "of_p_3_P", "of_p_4_P", "of_p_5_P", "of_p_6_P", "of_p_7_P", "of_p_8_P", "of_p_9_P", "of_p_10_P", "of_p_11_P", "of_p_12_P", "of_p_13_P", "of_p_14_P", "of_p_15_P", "of_p_16_P", "of_p_17_P", "of_p_18_P", "of_p_19_P", "of_p_20_P", "of_p_21_P", "of_p_22_P", "of_p_23_P", "of_p_24_P", "of_p_25_P", "of_p_26_P", "of_p_27_P", "of_p_28_P", "of_p_29_P", "of_p_30_P", "of_p_31_P", "of_p_32_P", "of_p_33_P"]
# 5. Facenet Embeddings
f5s = [f'fn_e{x}_{p}' for x in range(0, 128) for p in ['I', 'P']]


Y_cols = ['PID', 'Hirability_Interview', 'Evaluation Score_POST', 'Confident', 'Stressed', 'Uncomfortable']

# Get sub-dataframe with selected features and 'PID'
df_X_partial = df_master[f1s + f2s + f3s + f4s + ['PID']]
df_facenet = pd.read_csv('../data/processed/deepface/facenet/facenet_master.csv').drop(columns=['fn_face_confidence_I', 'fn_face_confidence_P', 'frame'])

del df_master

# Compute mean, variance, max, min, and standard deviation per PID
df_X_partial_agg = df_X_partial.groupby('PID').agg(['mean', 'var', 'max', 'min']).reset_index()

del df_X_partial

df_facenet_agg = df_facenet.groupby('PID').agg(['mean', 'var', 'max', 'min']).reset_index()

del df_facenet

# Concat X df with FaceNet df (Do after aggregation over interview because facenet embeddings are for every third frame)
df_X_agg = pd.concat([df_X_partial_agg, df_facenet_agg.drop(columns=['PID'])], axis=1)

# Flatten the columns
df_X_agg.columns = ['PID'] + ['{}_{}'.format(col[0], col[1]) for col in df_X_agg.columns[1:]]

# Define the statistical measures
# TODO: choose var or std
stats = ['mean', 'var', 'max', 'min']

# Create new feature lists with statistics
new_features = ['{}_{}'.format(feat, stat) for feat in f1s + f2s + f3s + f4s + f5s for stat in stats]

# Similarly, create feature lists for each feature group
f1s_stats = ['{}_{}'.format(feat, stat) for feat in f1s for stat in stats]
f2s_stats = ['{}_{}'.format(feat, stat) for feat in f2s for stat in stats]
f3s_stats = ['{}_{}'.format(feat, stat) for feat in f3s for stat in stats]
f4s_stats = ['{}_{}'.format(feat, stat) for feat in f4s for stat in stats]
f5s_stats = ['{}_{}'.format(feat, stat) for feat in f5s for stat in stats]

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
    
def generate_hyperparameters(base_params, num_features):
    dynamic_params = {}
    
    for key, value in base_params.items():
        if callable(value):
            dynamic_params[key] = value(num_features)
        else:
            dynamic_params[key] = value

    return dynamic_params


def eval_features(df_X, df_Y, X_features, Y_feature, model_class, base_hyper_params, is_regression=True):
    
    num_features = len(X_features)
    
    hyper_parameters = generate_hyperparameters(base_hyper_params, num_features)
    
    predictions = []
    actuals = []
    
    for cv_outer_i in tqdm(df_X.index, leave=False):
        cv_outer_x = df_X.iloc[[cv_outer_i]]
        cv_outer_y = df_Y.iloc[cv_outer_i][Y_feature]
        
        train_outer_x = df_X.drop(index=cv_outer_i).reset_index(drop=True)
        train_outer_y = df_Y.drop(index=cv_outer_i)[Y_feature].reset_index(drop=True)
        
        if is_regression:
            scoring = 'neg_mean_absolute_error'
        else:
            scoring = 'accuracy'
            
        loo_inner = LeaveOneOut()
        
        model = model_class(random_state=1)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=hyper_parameters,
            cv=loo_inner,
            scoring=scoring,
            # verbose=True,
            n_jobs=-1 # use all processors
        )
        
        try:
            grid_search.fit(train_outer_x, train_outer_y)
        except MemoryError as e:
            print('memory error:', e)
            return
        except Exception as e:
            print('error in grid search fit: e')
            return
            
        
        best_params = grid_search.best_params_
        # print(f'best params at cv {cv_outer_i}: {best_params} - score: {grid_search.best_score_}')
        
        # Re-train on whole inner cv
        best_model = model_class(**best_params, random_state=1)
        best_model.fit(train_outer_x, train_outer_y)
        
        # Predict and store
        pred = best_model.predict(cv_outer_x)
        predictions.append(pred)
        actuals.append(cv_outer_y)
        
    if is_regression:
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals) 
        
        # Graph Pred vs Actual
        plt.figure()
        plt.scatter(actuals, predictions)
        
        # Add y=x reference line
        line_min = min(min(actuals), min(predictions))
        line_max = max(max(actuals), max(predictions))
        plt.plot([line_min, line_max], [line_min, line_max], 'r--')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_class.__name__}')
        
        feature_class = 'ERROR'
        if 'of_gaze_0_x_I_mean' in X_features:
            feature_class = 'Eye Gaze'
        elif 'of_pose_Tx_I_mean' in X_features:
            feature_class = 'Face Pose'
        elif 'of_AU01_r_I_mean' in X_features:
            feature_class = 'Facial AUs'
        elif 'of_p_scale_I_mean' in X_features:
            feature_class = 'PDM'
        elif 'fn_e0_I_mean' in X_features:
            feature_class = 'Facenet Embeddings'
        
        plt.savefig(f'../reports/baselines/figures/{model_class.__name__}_{Y_feature}_{feature_class}.png')
        plt.close()
        
        final_mae = abs(predictions - actuals).mean()
        final_mse = ((predictions - actuals)**2).mean()
        final_rmse = np.sqrt(final_mse)
        pearsons = pearsonr(predictions, actuals)
        r2 = r2_score(predictions, actuals)
        print(f'\t\t{model_class.__name__:<23}, r: {pearsons[0]:.3f} p: {pearsons[1]:.4f} R^2: {r2:.3f} MAE: {final_mae:.3f} MSE: {final_mse:.3f} RMSE: {final_rmse:.3f}')
    else:
        bal_acc = balanced_accuracy_score(actuals, predictions)
        final_acc = (np.array(predictions) == np.array(actuals)).mean()
        macro_f1_score = f1_score(actuals, predictions, average='macro')
        print(f'\t\t{model_class.__name__:<23}, bal acc: {bal_acc:.3f}, acc: {final_acc:.3f}, macro f1: {macro_f1_score:.3f}')

class FFNN(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(64, 32), drop_out=0.1, regularization=0.0001, input_size=10, output_size=1, learning_rate=0.001, epochs=100, is_regression=True, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.drop_out = drop_out
        self.regularization = regularization
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_regression = is_regression
        self.random_state = random_state  # Adding random_state for reproducibility

        # If a random_state is provided, set the seed for reproducibility
        if random_state is not None:
            self.set_random_state(random_state)

        self.model = None
        self.criterion = None
        self.optimizer = None

    def set_random_state(self, random_state):
        np.random.seed(random_state)  # For numpy operations
        torch.manual_seed(random_state)  # For PyTorch operations (CPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)  # For PyTorch operations (GPU)
    

    def _build_model(self):
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_layer_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.drop_out))
        
        for i in range(len(self.hidden_layer_sizes) - 1):
            layers.append(nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.drop_out))
        
        layers.append(nn.Linear(self.hidden_layer_sizes[-1], self.output_size))
        
        self.model = nn.Sequential(*layers)
        if self.is_regression:
            self.criterion = nn.MSELoss()  # Assuming regression, switch to another criterion for classification
        else: 
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self, X, y):
        # Convert DataFrame to NumPy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Ensure correct types
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Assuming y is 1D for regression
        
        self._build_model()
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        # Convert DataFrame to NumPy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ensure correct types
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        if self.is_regression:
            return outputs.numpy().flatten()  # Return as a flat array for regression
        else:
            predictions = torch.sigmoid(outputs)
            return (predictions >= 0.5).numpy().astype(int).flatten() # Threshold at 0.5 to get binary output (0 or 1)

    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'drop_out': self.drop_out,
            'regularization': self.regularization,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class FFNNRegressor(FFNN, RegressorMixin):
    def __init__(self, **args):
        super().__init__(**args)
        self.is_regression = True
        
class FFNNClassifier(FFNN, ClassifierMixin):
    def __init__(self, **args):
        super().__init__(**args)
        self.is_regression = False

dt_hyper_params = {
    'max_depth': lambda n: [int(0.5*math.sqrt(n)), int(1*math.sqrt(n)), int(1.5*math.sqrt(n)), int(2*math.sqrt(n))],
    'min_samples_split': [2, 0.25, 0.5]
}

xgb_hyper_params = {
    'max_depth': lambda n: [int(0.5*math.sqrt(n)), int(1*math.sqrt(n)), int(1.5*math.sqrt(n)), int(2*math.sqrt(n))],
    'n_estimators': [20,30,50,100]
}

mlp_hyper_params = {
    'hidden_layer_sizes': lambda n: [(int(n/5), int(n/10)), (int(n/3), int(n/6)), (int(n/2), int(n/4)), (int(n), int(n/2))],
    'alpha': [0.0001, 0.0005, 0.001], # L2 Regularization
    'max_iter': [1000]
}

ffnn_hyper_params = {
    'hidden_layer_sizes': lambda n: [(int(n/3), int(n/6)), (int(n/2), int(n/4)), (int(n), int(n/2))],
    'drop_out': [0.1, 0.25, 0.5, 0.75],
    'regularization': [0.0001, 0.0005, 0.001],
    ## Not hyper params but just params that don't change
    'input_size': lambda n: [n],
}

hyper_params_list = [dt_hyper_params, xgb_hyper_params, mlp_hyper_params] #, ffnn_hyper_params]
regression_models = [tree.DecisionTreeRegressor, xgb.XGBRegressor, MLPRegressor] #, FFNNRegressor]
classification_models = [tree.DecisionTreeClassifier, xgb.XGBClassifier, MLPClassifier] #, FFNNClassifier]

for feature_set in [f1s_stats, f2s_stats, f3s_stats, f4s_stats, f5s_stats]:
    print(f'Feature set: {feature_set}')
    for y_feat in Y_cols[1:]:
        print(f'\tPrediction Feature: {y_feat}')
        for model, hp in list(zip(regression_models, hyper_params_list)):
            eval_features(df_X_scaled[feature_set], df_Y_scaled, feature_set, y_feat, model, hp, is_regression=True)
        for model, hp in list(zip(classification_models, hyper_params_list)):
            eval_features(df_X_scaled[feature_set], df_Y_bin, feature_set, y_feat, model, hp, is_regression=False)


eval_features(df_X_scaled[f1s_stats], df_Y_scaled, f1s_stats, Y_cols[1], tree.DecisionTreeRegressor, dt_hyper_params, is_regression=True)
eval_features(df_X_scaled[f1s_stats], df_Y_scaled, f1s_stats, Y_cols[1], xgb.XGBRegressor, xgb_hyper_params, is_regression=True)
eval_features(df_X_scaled[f1s_stats], df_Y_scaled, f1s_stats, Y_cols[1], MLPRegressor, mlp_hyper_params, is_regression=True)
eval_features(df_X_scaled[f1s_stats], df_Y_scaled, f1s_stats, Y_cols[1], FFNNRegressor, ffnn_hyper_params, is_regression=True)

eval_features(df_X_scaled[f1s_stats], df_Y_bin, f1s_stats, Y_cols[1], tree.DecisionTreeClassifier, dt_hyper_params, is_regression=False)
eval_features(df_X_scaled[f1s_stats], df_Y_bin, f1s_stats, Y_cols[1], xgb.XGBClassifier, xgb_hyper_params, is_regression=False)
eval_features(df_X_scaled[f1s_stats], df_Y_bin, f1s_stats, Y_cols[1], MLPClassifier, mlp_hyper_params, is_regression=False)
eval_features(df_X_scaled[f1s_stats], df_Y_bin, f1s_stats, Y_cols[1], FFNNClassifier, ffnn_hyper_params, is_regression=False)