# -*- coding: utf-8 -*-
"""
@author: vasiliskv
"""
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt

from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier, IsolationForest
from sklearn.model_selection import  StratifiedKFold, GridSearchCV, cross_validate
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve


# Calculate and plot the feature correlation matrix 
# Find the highly correlated features (corr_coef>0.98) and drop the first in order
def find_corr_columns(features):
    corr_matrix = features.corr()
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
   
    with sns.axes_style("white"):
        f1, ax1 = plt.subplots()
        ax1 = sns.heatmap(corr_matrix, mask=mask, square=True, linewidths=.01, cmap="RdBu")
        
    f2, ax2 = plt.subplots()
    ax2 = sns.heatmap(corr_matrix, linewidths=.01, cmap="RdBu")
    plt.show() 
   
    corr_columns = []
    opp_mask = mask==0
    masked_corr_matrix = corr_matrix*opp_mask
    for i in range(masked_corr_matrix.shape[1]):
        if (abs(masked_corr_matrix.iloc[:,i])>=0.98).any():
            corr_columns.append(i)
    return corr_columns


# 0-1 Normalize the features
def normalize_train_test(train_set,  test_set, val_set=None):
    norm = MinMaxScaler().fit(train_set)
    norm_x_train = pd.DataFrame(norm.transform(train_set), columns=train_set.columns)
    norm_X_test = pd.DataFrame(norm.transform(test_set), columns=train_set.columns)
    if type(val_set) !=  "NoneType":
        norm_x_val = pd.DataFrame(norm.transform(val_set), columns=train_set.columns)
    else:
        norm_x_val = None

    return norm_x_train, norm_x_val, norm_X_test


# Find the best features for pre-defined classifiers , using CrossValidation for safer results
def find_best_features(x, y, selected_feat_lists, models):
    
    models_best_features = {}
    feat_help_l = ["all", "l_mutual_inf", "l_rfecv", "l_feat_imp"]
    
    for model_name in models.keys():
        print("model: "+ model_name)
        scores_l = []
        for selected_feat in selected_feat_lists.keys():
            cv_scores = cross_validate(models[model_name], x.loc[:,selected_feat_lists[selected_feat]], 
                                       y, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=4)
            scores_l.append(cv_scores['test_score'].mean())
        print(scores_l)
        best_feats_name = feat_help_l[np.array(scores_l).argmax()]
        print(best_feats_name)
        print()
        models_best_features[model_name] = selected_feat_lists[best_feats_name]
        
    return models_best_features


# Find the best hyperparameters for pre-defined supervised classifiers (Statified 5)
def superv_hyperparameter_tuning(norm_x_train, y_train, models_best_features):
    grid_search_res = {}
    grid_search_best_est = {}
    grid_search_best_score = {}
    
    models = {"LG": LogisticRegression(max_iter = 400), "SVM": SVC(), "RF": RandomForestClassifier(), 
              "KNN": KNeighborsClassifier()}
    
    hyperparameters_grids = {
        "LG": [{"class_weight": [None, "balanced"], 
                "solver": ["lbfgs", "liblinear"], 
                "C": [0.1, 1, 10, 100], 
                "tol": [1e-5, 1e-4, 1e-3]
                }], 
        "SVM": [{"class_weight": [None, "balanced", {1: 5}, {1: 10}], 
                 "kernel": ["linear", "poly", "rbf"], 
                 "gamma": [0.01, 0.001, 0.0001], 
                 "C": [1, 10, 100, 1000]
                 }],
        "RF": [{"n_estimators": [50, 100, 150, 200],
                "min_samples_split": [2, 3, 4, 5],
                "max_features": ["sqrt","log2",3,7,12,15,20]
                }],
        "KNN": [{"n_neighbors": [i for i in range(1,20)], 
                 "weights": ["uniform", "distance"]
                 }],
        }
    
    for model_name in models.keys():
        print("Starting GridSearch for " + model_name)
        
        grid_search = GridSearchCV(models[model_name], hyperparameters_grids[model_name], 
                                   cv=StratifiedKFold(5), n_jobs=4, scoring="roc_auc")
    
        grid_search.fit(norm_x_train.loc[:,models_best_features[model_name]], y_train)
        
        grid_search_res[model_name] = grid_search.cv_results_
        grid_search_best_est[model_name] = grid_search.best_estimator_
        grid_search_best_score[model_name] = grid_search.best_score_
        
    return grid_search_res, grid_search_best_est, grid_search_best_score


# Find the best hyperparameters for pre-defined unsupervised classifiers (Statified 5)
def unsuperv_hyperparameter_tuning(x, y, models_best_features):
    grid_search_res = {}
    grid_search_best_est = {}
    grid_search_best_score = {}
    
    models = {"IF": IsolationForest(), "O_SVM": OneClassSVM()}
    
    # Hyperparameter tuning
    hyperparameters_grids  = {
        "IF": [{"n_estimators": [100, 150, 200],
                "max_samples": [30, 50, 100, 150, 200, "auto"],
                "contamination": [0.07, 0.1, 0.15, 0.2, 0.25, "auto"],
                "max_features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            }],
        "O_SVM": [{
                "nu": [0.1, 0.15, 0.2, 0.25, 0.3],
                "gamma": ['scale', 'auto', 0.01, 0.001, 0.0001]
                }]
        }
    
    for model_name in models.keys():
        print("Starting GridSearch for " + model_name)
        
        grid_search = GridSearchCV(models[model_name], hyperparameters_grids[model_name], 
                                   cv=StratifiedKFold(5), n_jobs=4, scoring="roc_auc")
    
        grid_search.fit(x.loc[:,models_best_features[model_name]], y)
        
        grid_search_res[model_name] = grid_search.cv_results_
        grid_search_best_est[model_name] = grid_search.best_estimator_
        grid_search_best_score[model_name] = grid_search.best_score_
        
    return grid_search_res, grid_search_best_est, grid_search_best_score


# Plot the ROC curve
def plot_roc(y_test, l_y_score, estim_name_list):
    plt.figure()
    for estim_name, y_score in zip(estim_name_list, l_y_score):
        
        fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=True)
        lw = 2.1
        plt.plot(fpr, tpr, lw=lw, label=estim_name + ' roc-auc = %0.2f' % roc_auc_score(y_test, y_score))
        # plt.plot([0, 1], [0, 1], color='darkred', lw=lw, linestyle='--')
    
    plt.xlim([-0.005, 1.005])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc="lower right", fontsize=14)
    plt.title("Receiver Operating Characteristic curves", fontsize=18)
    
    
# Change labels to -1,1
def change_labels(old_y):
    new_y = deepcopy(old_y)
    
    for i in range(len(old_y)):
        if old_y.iloc[i]==True:
            new_y.iloc[i]=-1
        else:
            new_y.iloc[i]=1
    new_y = np.array(new_y, dtype="int32")
    return new_y


# Alter features lists for Unsupervised models
def fix_feat_lists(non_corr_feats, selected_feat_lists):
    un_l_mutual_inf = [i for i in selected_feat_lists["l_mutual_inf"] if i in non_corr_feats]
    un_l_rfecv = [i for i in selected_feat_lists["l_rfecv"] if i in non_corr_feats]
    un_l_feat_imp = [i for i in selected_feat_lists["l_feat_imp"] if i in non_corr_feats]
    
    return {"all": non_corr_feats,
            "l_mutual_inf": un_l_mutual_inf, 
            "l_rfecv": un_l_rfecv, 
            "l_feat_imp": un_l_feat_imp
            }


# Feature Selection using three methods Mutaul info, RFE-CV (LR) and Feature importances (RandomForestClassifier)
def feature_selection(norm_x_train, y_train):
    
    # Select K best with k=#feats/2
    print("mutual info selectKbest")
    __num_of_feats_to_sel = norm_x_train.shape[1]//2
    selector = SelectKBest(mutual_info_classif, k=__num_of_feats_to_sel)
    
    selector.fit(norm_x_train, y_train)
    
    l_mutual_inf = norm_x_train.columns[selector.get_support()].tolist()
    print("Optimal number of features : %d" % len(l_mutual_inf))
    print(l_mutual_inf)
    
    
    # RFE (greedy) with cross-validation. Logistic regression used
    
    min_features_to_select = 1
    
    print("RFECV")
    __est = LogisticRegression(solver="liblinear")
    rfecv = RFECV(estimator=__est, step=1, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=4, min_features_to_select = min_features_to_select)
    rfecv.fit(norm_x_train, y_train)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    l_rfecv = norm_x_train.columns[rfecv.get_support()].tolist()
    print(l_rfecv)
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select), rfecv.grid_scores_)
    plt.show()
    
    # Feature importances using RandomForestClassifier
    print("feature importances")
    
    __est = RandomForestClassifier()
    featureSelection = SelectFromModel(__est)
    featureSelection.fit(norm_x_train, y_train)
    
    l_feat_imp = norm_x_train.columns[featureSelection.get_support()].tolist()
    print("Optimal number of features : %d" % len(l_feat_imp))
    print(l_feat_imp)   
    
    return {"all": norm_x_train.columns.tolist(), "l_mutual_inf": l_mutual_inf, "l_rfecv": l_rfecv, "l_feat_imp": l_feat_imp}


def evaluate_estimators(estimators, models_best_features, X_test, y_test):
    eval_columns = ["accuracy", "recall", "precision", "f1_score", "matthews_cc", "roc_auc"]
    eval_indexes = list(estimators.keys())
    eval_values = []
    l_y_score = []
    
    for estim_name in estimators.keys():
        y_pred = estimators[estim_name].predict(X_test.loc[:,models_best_features[estim_name]])
        if estim_name == "IF" or estim_name == "O_SVM":
            y_score = estimators[estim_name].score_samples(X_test.loc[:,models_best_features[estim_name]])
        else:
            y_score = estimators[estim_name].predict_proba(X_test.loc[:,models_best_features[estim_name]])[:, 1]
        eval_values.append([accuracy_score(y_test, y_pred), 
                        recall_score(y_test, y_pred), 
                        precision_score(y_test, y_pred), 
                        f1_score(y_test, y_pred), 
                        matthews_corrcoef(y_test, y_pred), 
                        roc_auc_score(y_test, y_score)])
        
        print(estim_name)
        print([accuracy_score(y_test, y_pred), 
            recall_score(y_test, y_pred, pos_label=1), 
            precision_score(y_test, y_pred, pos_label=1), 
            f1_score(y_test, y_pred, pos_label=1), 
            roc_auc_score(y_test, y_score)])
        print(confusion_matrix(y_test, y_pred))

        l_y_score.append(y_score)
        
    return l_y_score, pd.DataFrame(eval_values, index=eval_indexes, columns=eval_columns)
        
    
    