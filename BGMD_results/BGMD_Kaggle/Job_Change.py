from common_imports import *

def BGMD_job_change():
    # Reading the dataset
    data = pd.read_csv("dataset/Job_Change/aug_train.csv")
    aug_train = data.sample(frac=1, replace=True, random_state=1).reset_index(drop=True)

    # Seperate aug_train into target and features 
    y = aug_train['target']
    X_aug_train = aug_train.drop('target',axis = 'columns')
    # save the index for X_aug_train 
    X_aug_train_index = X_aug_train.index.to_list()

    class MultiColumnLabelEncoder:
        def __init__(self,columns = None):
            self.columns = columns # array of column names to encode

        def fit(self,X,y=None):
            return self # not relevant here

        def transform(self,X):
            '''
            Transforms columns of X specified in self.columns using
            LabelEncoder(). If no columns specified, transforms all
            columns in X.
            '''
            output = X.copy()
            if self.columns is not None:
                for col in self.columns:
                    # convert float NaN --> string NaN
                    output[col] = output[col].fillna('NaN')
                    output[col] = LabelEncoder().fit_transform(output[col])
            else:
                for colname,col in output.iteritems():
                    output[colname] = LabelEncoder().fit_transform(col)
            return output

        def fit_transform(self,X,y=None):
            return self.fit(X,y).transform(X)

    # store the catagorical features names as a list      
    cat_features = X_aug_train.select_dtypes(['object']).columns.to_list()

    # use MultiColumnLabelEncoder to apply LabelEncoding on cat_features 
    # uses NaN as a value , no imputation will be used for missing data
    X = MultiColumnLabelEncoder(columns = cat_features).fit_transform(X_aug_train)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

    print("Train data: ", X_train.shape)
    print("Test data: ", X_test.shape)
    
    ##############
    
    y_train.value_counts()
    
    ##############
    
    y_test.value_counts()
    
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