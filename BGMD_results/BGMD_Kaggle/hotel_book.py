from common_imports import *

def BGMD_hotel_book():
    # Reading the dataset
    raw_train = pd.read_csv("dataset/tabular/train.csv")
    #raw_train = raw_train.sample(frac=0.01, replace=True, random_state=1)
    target = raw_train.target
    X_train, X_test, y_train, y_test = train_test_split(raw_train, target, test_size = 0.5, random_state = 29)

    X_train = X_train.sample(frac=0.03, replace=True, random_state=1).reset_index(drop=True)
    target = X_train.target
    X_train = X_train.drop('target', axis = 1)
    train = X_train.drop('id', axis = 1)

    X_test = X_test.sample(frac=0.03, replace=True, random_state=1).reset_index(drop=True)
    y_test = X_test.target
    X_test = X_test.drop('target', axis = 1)
    test = X_test.drop('id', axis = 1)

    print("Train data: ", train.shape)
    print("Test data: ", test.shape)
    
    ##############
    
    X_test = test
    
    ## Default Model
    
    model_default = svm.SVC()
    scores_default = cross_val_score(model_default, X=train, y=target, cv = FOLDS)
    model_default.fit(train, target)
    y_pred_default = model_default.predict(X_test)
    get_performance(X_test, y_test, y_pred_default)
    
    ## MAPS
    
    default_result = pd.concat([X_test, y_test], axis=1, join='inner')
    default_result.loc[:,"pred"] = y_pred_default
    
    ##############
    
    def mispredict_label(row):
        if row['target'] == row['pred']:
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
    scores_default = cross_val_score(model_default, X=train, y=target, cv = FOLDS)
    model_default.fit(train, target)
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