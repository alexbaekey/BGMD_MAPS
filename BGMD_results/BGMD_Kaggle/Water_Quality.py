from common_imports import *

def BGMD_water_quality():
    data_dir = "dataset/Water_Quality/"
    data = pd.read_csv(data_dir + "water_potability.csv")
    data = data.sample(frac=1, replace=True, random_state=1).reset_index(drop=True)
    data = data.dropna()
    label = data.columns[-1]
    features = data.columns[:-1]
    # Separate the data
    X, y = data[features], data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    ##############
    
    y_train.value_counts()
    
    ##############
    
    y_test.value_counts()
    
    ##############
    
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    ## Default Model
    
    model_default = svm.SVC(kernel='sigmoid')
    scores_default = cross_val_score(model_default, X=X_train, y=y_train, cv = FOLDS)
    model_default.fit(X_train, y_train)
    y_pred_default = model_default.predict(X_test)
    get_performance(X_test, y_test, y_pred_default)
    
    ##############
    
    pd.DataFrame(y_pred_default).value_counts()
    
    ##############
    
    default_result = pd.concat([X_test, y_test], axis=1, join='inner')
    default_result.loc[:,"pred"] = y_pred_default
    
    ##############
    
    def mispredict_label(row):
        if row['Potability'] == row['pred']:
            return False
        return True
    
    ##############
    
    default_result_copy = default_result.copy()
    X_test_copy = X_test.copy()
    X_test_copy['mispredict'] = default_result_copy.apply(lambda row: mispredict_label(row), axis=1)
    
    ##############
    
    settings = diagnoser.Settings
    settings.all_rules = True
    # Get relevent attributes and target 
    relevant_attributes, Target = get_relevent_attributs_target(X_test_copy)
    # Generate MMD rules and correspodning information
    MMD_rules, MMD_time, MMD_Features = get_MMD_results(X_test_copy, relevant_attributes, Target)

    #Get biased attributes this time 
    biased_attributes = get_biased_features(X_test_copy, relevant_attributes)

    BGMD_rules, BGMD_time, BGMD_Features = get_BGMD_results(X_test_copy, biased_attributes, Target)

    print('MMD Spent:', MMD_time, 'BGMD Spent:', BGMD_time)
    MMD_rules, BGMD_rules
    
    ## Decision Tree
    
    model_default = DecisionTreeClassifier()
    scores_default = cross_val_score(model_default, X=X_train, y=y_train, cv = FOLDS)
    model_default.fit(X_train, y_train)
    y_pred_default = model_default.predict(X_test)
    get_performance(X_test, y_test, y_pred_default)
    
    ##############
    
    default_result = pd.concat([X_test, y_test], axis=1, join='inner')
    default_result.loc[:,"pred"] = y_pred_default
    default_result_copy = default_result.copy()
    X_test_copy = X_test.copy()
    X_test_copy['mispredict'] = default_result_copy.apply(lambda row: mispredict_label(row), axis=1)
    settings = diagnoser.Settings
    settings.all_rules = True
    # Get relevent attributes and target 
    relevant_attributes, Target = get_relevent_attributs_target(X_test_copy)
    # Generate MMD rules and correspodning information
    MMD_rules, MMD_time, MMD_Features = get_MMD_results(X_test_copy, relevant_attributes, Target)

    #Get biased attributes this time 
    biased_attributes = get_biased_features(X_test_copy, relevant_attributes)

    BGMD_rules, BGMD_time, BGMD_Features = get_BGMD_results(X_test_copy, biased_attributes, Target)

    print('MMD Spent:', MMD_time, 'BGMD Spent:', BGMD_time)
    MMD_rules, BGMD_rules
    
    