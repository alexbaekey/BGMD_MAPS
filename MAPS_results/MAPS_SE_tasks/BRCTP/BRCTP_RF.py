from collections import Counter
from helper import generateTrain_data_Weights
from helper import generate_JTT_Weights
from helper import get_performance, get_conflict_commit_performance, get_merge_coflict_mispredicted_region_test
from helper import get_test_data_in_misprediction_areas
from helper import get_top_f1_rules, get_relevent_attributs_target, get_MMD_results, get_biased_features, get_BGMD_results
from imblearn.over_sampling import SMOTE
from mmd import diagnoser
from scipy import stats as st
from sklearn import svm
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import sys
import time 
import xgboost as xgb


def BRCTP_RF():
    EPOCHS = 700
    
    BATCH_SIZE = 2048
    
    ACTIVATION = 'swish'
    
    LEARNING_RATE = 0.0007
    
    FOLDS = 5
    
    TRAIN_PATH_P2 = "/data/BRCTP/p2_train.csv"
    
    TEST_PATH_P2 = "/data/BRCTP/p2_test.csv"
    
    TRAIN_PATH_P3 = "/data/BRCTP/p3_train.csv"
    
    TEST_PATH_P3 = "/data/BRCTP/p3_test.csv"
    
    TRAIN_PATH_P4 = "/data/BRCTP/p4_train.csv"
    
    TEST_PATH_P4 = "/data/BRCTP/p4_test.csv"
    
    
    
    train_p2 = pd.read_csv(TRAIN_PATH_P2)
    
    test_p2 = pd.read_csv(TEST_PATH_P2)
    
    train_p3 = pd.read_csv(TRAIN_PATH_P3)
    
    test_p3 = pd.read_csv(TEST_PATH_P3)
    
    train_p4 = pd.read_csv(TRAIN_PATH_P4)
    
    test_p4 = pd.read_csv(TEST_PATH_P4)
    
    
    
    all_data = pd.concat([train_p2, train_p3, train_p4, test_p2, test_p3, test_p4])
    
    
    
    all_data = all_data.sample(frac=1, random_state=1).reset_index(drop=True)
    
    
    
    X = all_data.drop(['OOSLA'], axis=1)
    
    y = all_data[['OOSLA']]
    
    
    
    print(X.shape)
    
    print(y.shape)
    
    print('------')
    
    print(y.value_counts())
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model_default = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)
    
    scores_default = cross_val_score(model_default, X=X_train, y=y_train, cv = FOLDS)
    
    model_default.fit(X_train, y_train)
    
    y_pred_default = model_default.predict(X_test)
    
    
    
    print(classification_report(y_test, y_pred_default))
    
    # SMOTE Oversampling
    
        # Transform data 
    
    oversample = SMOTE()
    
    X_train_sm, y_train_sm = oversample.fit_resample(X_train, y_train)
    
    model_SMOTE = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)
    
    scores_SMOTE = cross_val_score(model_SMOTE, X=X_train_sm, y=y_train_sm, cv = FOLDS)
    
    model_SMOTE.fit(X_train_sm, y_train_sm)
    
    y_pred_SMOTE = model_SMOTE.predict(X_test)
    
    
    
    print(classification_report(y_test, y_pred_SMOTE))
    
    y_train_sm
    
    sys.path.insert(1, './mmd')
    
    #notebook's library
    
    def mispredict_label(row):
    
        if row['OOSLA'] == row['pred']:
    
            return False
    
        return True
    
    #use first default model result to detect mispredictions
    
    default_result = pd.concat([X_test, y_test], axis=1, join='inner')
    
    default_result.loc[:,"pred"] = y_pred_default
    
    
    
    # default_result = default_result.loc[default_result["is_conflict"] == 1]
    
    default_result_copy = default_result.copy()
    
    X_test_copy = default_result.copy().drop(['OOSLA', 'pred'], axis=1)
    
    X_test_copy['mispredict'] = default_result_copy.apply(lambda row: mispredict_label(row), axis=1)
    
    settings = diagnoser.Settings
    
    settings.all_rules = False
    
    
    
    # Get relevent attributes and target 
    
    relevant_attributes, Target = get_relevent_attributs_target(X_test_copy)
    
    # Geerate MMD rules and correspodning information
    
    MMD_rules, MMD_time, MMD_Features = get_MMD_results(X_test_copy, relevant_attributes, Target, settings)
    
    
    
    #Get biased attributes this time 
    
    biased_attributes = get_biased_features(X_test_copy, relevant_attributes)
    
    
    
    BGMD_rules, BGMD_time, BGMD_Features = get_BGMD_results(X_test_copy, biased_attributes, Target, settings)
    
    
    
    
    
    print('MMD Spent:', MMD_time, 'BGMD Spent:', BGMD_time)
    
    MMD_rules, BGMD_rules
    
    
    
    final_result = pd.concat([X_test, y_test], axis=1, join='inner')
    
    
    
    indexes_in_misprediction_area = get_test_data_in_misprediction_areas(BGMD_rules, X_test)
    
    y_actual_MD = []
    
    
    
    for index in indexes_in_misprediction_area:
    
            y_actual_MD.append(final_result.loc[index]['OOSLA'])
    
    grid_upweights = range(1, 21)
    
    best_weight_all = 1
    
    best_f1_all = 0
    
    best_weight_mis = 1
    
    best_f1_mis = 0
    
    f1_all = []
    
    f1_mispredicted = []
    
    acc_all = []
    
    acc_mispredicted = []
    
    recall_all = []
    
    recall_mispredicted = []
    
    for upweight in grid_upweights:
    
        # give extra weights to training samples in mispredited areas 
    
        print('Upweight_value: ', upweight)
    
        weights = generateTrain_data_Weights(BGMD_rules, X_train, upweight_value=upweight)
    
    
    
        c = Counter(weights)
    
    
    
        MAPS_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)
    
        scores_MAPS = cross_val_score(MAPS_model, X=X_train, y=y_train, cv = FOLDS)
    
        MAPS_model.fit(X_train, y_train, sample_weight = weights)
    
    
    
        y_pred_MAPS = MAPS_model.predict(X_test)
    
    
    
        total_result = get_conflict_commit_performance(X_test, y_test, y_pred_MAPS)
    
        
    
        acc_all.append(total_result[0][0])
    
        recall_all.append(total_result[1][0])
    
        f1_all.append(total_result[2][0])
    
        
    
        final_result['y_pred_MAPS'] = y_pred_MAPS
    
        y_pred_MAPS_MD = []
    
        for index in indexes_in_misprediction_area:
    
            y_pred_MAPS_MD.append(final_result.loc[index]['y_pred_MAPS'])
    
            
    
        MAPS_MD_metric = precision_recall_fscore_support(y_actual_MD, y_pred_MAPS_MD, labels=[1])
    
        
    
        acc_mispredicted.append(MAPS_MD_metric[0][0])
    
        recall_mispredicted.append(MAPS_MD_metric[1][0])
    
        f1_mispredicted.append(MAPS_MD_metric[2][0])
    
        
    
        if total_result[2][0] > best_f1_all:
    
            best_f1_all = total_result[2][0]
    
            best_weight_all = upweight;
    
        if MAPS_MD_metric[2][0] > best_f1_mis:
    
            best_f1_mis = MAPS_MD_metric[2][0]
    
            best_weight_mis = upweight
    
    print("Best weight on all is: ", best_weight_all)
    
    print("Best weight on misprediction area is: ", best_weight_mis)
    
    plt.plot(grid_upweights, f1_all, label = "f1 on all data")
    
    plt.plot(grid_upweights, f1_mispredicted, label = "f1 on mispredicted area")
    
    plt.legend()
    
    plt.show()
    
    dict = {'acc_all': acc_all, 'recall_all': recall_all, 'f1_all': f1_all, 'acc_mispredicted': acc_mispredicted, 'recall_mispredicted': recall_mispredicted, 'f1_mispredicted': f1_mispredicted}
    
    out = pd.DataFrame(dict)
    
    # out.to_csv('Merge_Conflict_Java_DT.csv')
    
    # MAPS all
    
    acc_all[best_weight_all-1], recall_all[best_weight_all-1], f1_all[best_weight_all-1]
    
    # MAPS mispredict area
    
    acc_mispredicted[best_weight_mis-1], recall_mispredicted[best_weight_mis-1], f1_mispredicted[best_weight_mis-1]
    
    get_merge_coflict_mispredicted_region_test(X_test, y_test, y_pred_default, y_pred_SMOTE, y_pred_MAPS, BGMD_rules, ylabel='OOSLA')
    
    X_train_1st, X_val, y_train_1st, y_val = train_test_split(X_train, y_train, test_size = 0.33, random_state = 29)
    
    
    
    model_JTT = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)
    
    scores_JTT_1st = cross_val_score(model_JTT, X_train_1st, y=y_train_1st, cv = FOLDS)
    
    model_JTT.fit(X_train_1st, y_train_1st)
    
    y_pred_JTT_val = model_JTT.predict(X_val)
    
    y_val
    
    grid_upweights_JTT = [10, 20, 30, 40, 50]
    
    best_weight_all_JTT = 1
    
    best_f1_all_JTT = 0
    
    best_weight_mis_JTT = 1
    
    best_f1_mis_JTT = 0
    
    f1_all_JTT = []
    
    f1_mispredicted_JTT = []
    
    precision_all_JTT = []
    
    precision_mispredicted_JTT = []
    
    recall_all_JTT = []
    
    recall_mispredicted_JTT = []
    
    
    
    for i, upweight in enumerate(grid_upweights_JTT):
    
        weights = generate_JTT_Weights(y_val['OOSLA'].to_list(), y_pred_JTT_val, weight = upweight)
    
        
    
        #combine train and validate data
    
        weights = np.concatenate((np.ones(len(X_train_1st)), weights))
    
        y_train_2nd = np.concatenate((y_train_1st, y_val))
    
        X_train_2nd = pd.concat([X_train_1st, X_val])
    
        X_train_2nd['val_weight'] = weights
    
        X_train_2nd['target'] = y_train_2nd
    
        X_train_2nd = X_train_2nd.sample(frac=1, random_state=1).reset_index(drop=True)
    
        weights = X_train_2nd['val_weight'].to_list()
    
        y_train_2nd = X_train_2nd['target']
    
        X_train_2nd = X_train_2nd.drop(['val_weight', 'target'], axis=1)
    
        
    
        model_JTT = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)
    
        scores_JTT_2nd = cross_val_score(model_JTT, X_train_2nd, y=y_train_2nd, cv = FOLDS)
    
        model_JTT.fit(X_train_2nd, y_train_2nd, sample_weight = weights)
    
        y_pred_JTT = model_JTT.predict(X_test)
    
        
    
        total_result_JTT = get_conflict_commit_performance(X_test, y_test, y_pred_JTT)
    
        
    
        precision_all_JTT.append(total_result_JTT[0][0])
    
        recall_all_JTT.append(total_result_JTT[1][0])
    
        f1_all_JTT.append(total_result_JTT[2][0])
    
        
    
        final_result['y_pred_JTT'] = y_pred_JTT
    
        y_pred_JTT_MD = []
    
        for index in indexes_in_misprediction_area:
    
            y_pred_JTT_MD.append(final_result.loc[index]['y_pred_JTT'])
    
            
    
        JTT_MD_metric = precision_recall_fscore_support(y_actual_MD, y_pred_JTT_MD, labels=[1])
    
        
    
        precision_mispredicted_JTT.append(JTT_MD_metric[0][0])
    
        recall_mispredicted_JTT.append(JTT_MD_metric[1][0])
    
        f1_mispredicted_JTT.append(JTT_MD_metric[2][0])
    
        
    
        if total_result_JTT[2][0] > best_f1_all_JTT:
    
            best_f1_all = total_result_JTT[2][0]
    
            best_weight_all = i;
    
        if JTT_MD_metric[2][0] > best_f1_mis_JTT:
    
            best_f1_mis = JTT_MD_metric[2][0]
    
            best_weight_mis = i
    
    print("JTT Best weight on all is: ", grid_upweights_JTT[best_weight_all])
    
    print("JTT Best weight on mispredicted area is: ", grid_upweights_JTT[best_weight_mis])
    
    # JTT all
    
    precision_all_JTT[best_weight_all_JTT-1], recall_all_JTT[best_weight_all_JTT-1], f1_all_JTT[best_weight_all_JTT-1]
    
    # JTT mispredict area
    
    precision_mispredicted_JTT[best_weight_mis_JTT-1], recall_mispredicted_JTT[best_weight_mis_JTT-1], f1_mispredicted_JTT[best_weight_mis_JTT-1]