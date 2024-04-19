from common_imports import *

def BGMD_bank_marketing():
    data_dir = "dataset/Bank_Marketing/"
    data = pd.read_csv(data_dir + "bank-additional-full.csv", sep = ';')
    data = data.sample(frac=0.5, replace=True, random_state=1).reset_index(drop=True)

    data['y'].replace(['yes', 'no'], [0, 1], inplace=True)

    data['job'].replace(['housemaid' , 'services' , 'admin.' , 'blue-collar' , 'technician', 'retired' , 'management', 'unemployed', 'self-employed', 'unknown' , 'entrepreneur', 'student'] , [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

    data['education'].replace(['basic.4y' , 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown' , 'university.degree' , 'illiterate'], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)

    data['marital'].replace(['married', 'single', 'divorced', 'unknown'], [1, 2, 3, 4], inplace=True)

    data['default'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

    data['housing'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

    data['loan'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

    data['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

    labelencoder_X = LabelEncoder()
    data['contact']     = labelencoder_X.fit_transform(data['contact']) 
    data['month']       = labelencoder_X.fit_transform(data['month']) 
    data['day_of_week'] = labelencoder_X.fit_transform(data['day_of_week']) 

    data.rename(columns={'emp.var.rate' : 'emp_var_rate',
                        'cons.price.idx' : 'cons_price_idx',
                        'cons.conf.idx' : 'cons_conf_idx',
                        'nr.employed' : 'nr_employed'}, inplace=True)
    y = data['y']
    data = data.drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33)
    
    ##############
    
    y_train.value_counts()
    
    ##############
    
    y_test.value_counts()
    
    ##############
    
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    ##############
    
    ## Default Model
    
    model_default = svm.SVC()
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
        if row['y'] == row['pred']:
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