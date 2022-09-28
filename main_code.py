# -*- coding: utf-8 -*-
"""
@author: vasiliskv
"""

import helper_functions

import pandas as pd

from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

#%%
# Read csv files and normalize features

labels=pd.read_csv("../data/data/labels.csv", index_col=0)
features = pd.read_csv("../data/csvs/new_features.csv",header=None)
unlabeled_features = pd.read_csv("../data/csvs/new_un_features.csv",header=None)

labels = pd.DataFrame(labels.label, dtype="int32")
    
#%%

corr_columns = helper_functions.find_corr_columns(features)

non_corr_features = features.drop(columns=corr_columns)

#%%
# Split into train-Test and Normalize the features

X_train, X_test, y_train, y_test = train_test_split(non_corr_features, labels.label, test_size=0.25, 
                                                    random_state=42, stratify=labels.label)

norm_x_train, _,  norm_X_test = helper_functions.normalize_train_test(X_train, X_test)


#%%
# Selection of the best feature subset for each of the 4 classifiers, using CrossValidation for safer results 

selected_feat_lists = helper_functions.feature_selection(norm_x_train, y_train)
    
models = {"LG": LogisticRegression(), "SVM": SVC(), "RF": RandomForestClassifier(random_state=42), "KNN": KNeighborsClassifier()}
    
models_best_features = helper_functions.find_best_features(norm_x_train, y_train, selected_feat_lists, models)


#%%
# Grid Search for hyperparameter tuning using cross-validation (Statified 5)

grid_search_res, grid_search_best_est, grid_search_best_score = helper_functions.superv_hyperparameter_tuning(norm_x_train, y_train, models_best_features)


#%%

# Evaluate the best classifiers on the test data
grid_search_best_est["SVM"].probability=True
grid_search_best_est["RF"].random_state=0

l_y_score = []

for estim_name in grid_search_best_est.keys():
    grid_search_best_est[estim_name].fit(norm_x_train.loc[:,models_best_features[estim_name]], y_train)
    
l_y_score, eval_dataframe_supervised = helper_functions.evaluate_estimators(grid_search_best_est, models_best_features, norm_X_test, y_test)
    
helper_functions.plot_roc(y_test, l_y_score, grid_search_best_est.keys())

#%% 

# Prepare data, selected features and hyperparameters for Isolation Forest evaluation

corr_columns = helper_functions.find_corr_columns(unlabeled_features)

non_corr_unlab_features = unlabeled_features.drop(columns=corr_columns)
non_corr_features = features.drop(columns=corr_columns)

# Change labels to -1,1
new_y_test = helper_functions.change_labels(labels.label)

# Normalize the features
norm_unlabeled_features, _,  norm_labeled_features = helper_functions.normalize_train_test(non_corr_unlab_features, non_corr_features)

estimators = {"IF": IsolationForest(random_state=42), "O_SVM": OneClassSVM()}
best_features = {"IF": non_corr_features.columns.tolist() ,"O_SVM":non_corr_features.columns.tolist()}


for estim_name in estimators.keys():
    estimators[estim_name].fit(norm_unlabeled_features)
  
l_y_score, eval_dataframe_unsupervised = helper_functions.evaluate_estimators(estimators, best_features, norm_labeled_features, new_y_test)

helper_functions.plot_roc(new_y_test, l_y_score, estimators.keys())

#%%

# Selecting the best features and hyperparameters based on a part of the labeled data that is used only for validation
# Prepare data, selected features and hyperparameters for Isolation Forest evaluation

corr_columns = helper_functions.find_corr_columns(unlabeled_features)

non_corr_unlab_features = unlabeled_features.drop(columns=corr_columns)
non_corr_features = features.drop(columns=corr_columns)

X_val, X_test, y_val, y_test = train_test_split(non_corr_features, labels.label, test_size=0.5, random_state=42, stratify=labels.label)

# Change labels to -1,1
new_y_val = helper_functions.change_labels(y_val)

new_y_test = helper_functions.change_labels(y_test)

# Normalize the features
norm_unlabeled_features, norm_labeled_val,  norm_labeled_test = helper_functions.normalize_train_test(non_corr_unlab_features,  X_test, X_val)


# Find best features for isolation forest
estimators = {"IF": IsolationForest(), "O_SVM": OneClassSVM()}

print("best feature list selection")
selected_feat_lists = helper_functions.fix_feat_lists(non_corr_unlab_features.columns.tolist(), selected_feat_lists)


models_best_features = helper_functions.find_best_features(norm_labeled_val, new_y_val, selected_feat_lists, estimators)

print("hyperparameter tuning")
grid_search_res, grid_search_best_est, grid_search_best_score = helper_functions.unsuperv_hyperparameter_tuning(norm_labeled_val, y_val, models_best_features)

for estim_name in grid_search_best_est.keys():
    grid_search_best_est[estim_name].fit(norm_unlabeled_features.loc[:,models_best_features[estim_name]])

l_y_score, eval_dataframe_supervised = helper_functions.evaluate_estimators(grid_search_best_est, models_best_features, norm_labeled_test, new_y_test)
helper_functions.plot_roc(new_y_test, l_y_score, grid_search_best_est.keys())    
