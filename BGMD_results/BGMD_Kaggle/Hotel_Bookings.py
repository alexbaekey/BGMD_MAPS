from common_imports import *

def BGMD_hotel_bookings():
    
    # Reading the dataset
    data = pd.read_csv("dataset/Hotel_Booking/hotel_bookings.csv")
    data = data.sample(frac=0.2, replace=True, random_state=1).reset_index(drop=True)

    data = data.drop(['company'], axis = 1)
    data['children'] = data['children'].fillna(0)
    data['hotel'] = data['hotel'].map({'Resort Hotel':0, 'City Hotel':1})

    data['arrival_date_month'] = data['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                                'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
    def family(data):
        if ((data['adults'] > 0) & (data['children'] > 0)):
            val = 1
        elif ((data['adults'] > 0) & (data['babies'] > 0)):
            val = 1
        else:
            val = 0
        return val

    def deposit(data):
        if ((data['deposit_type'] == 'No Deposit') | (data['deposit_type'] == 'Refundable')):
            return 0
        else:
            return 1
        
    def feature(data):
        data["is_family"] = data.apply(family, axis = 1)
        data["total_customer"] = data["adults"] + data["children"] + data["babies"]
        data["deposit_given"] = data.apply(deposit, axis=1)
        data["total_nights"] = data["stays_in_weekend_nights"]+ data["stays_in_week_nights"]
        return data

    data = feature(data)
    # Information of these columns is also inside of new features, so it is better to drop them.
    # I did not drop stays_nights features, I can't decide which feature is more important there.
    data = data.drop(columns = ['adults', 'babies', 'children', 'deposit_type', 'reservation_status_date'])

    indices = data.loc[pd.isna(data["country"]), :].index 
    data = data.drop(data.index[indices])   
    data = data.drop(columns = ['arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'agent'], axis = 1)

    df1 = data.copy()
    #one-hot-encoding
    df1 = pd.get_dummies(data = df1, columns = ['meal', 'market_segment', 'distribution_channel',
                                                'reserved_room_type', 'assigned_room_type', 'customer_type', 'reservation_status'])
    le = LabelEncoder()
    df1['country'] = le.fit_transform(df1['country']) 
    # There are more than 300 classes, so I wanted to use label encoder on this feature.

    df2 = df1.drop(columns = ['reservation_status_Canceled', 'reservation_status_Check-Out', 'reservation_status_No-Show'], axis = 1)
    df2.rename(columns={'market_segment_Offline TA/TO' : 'market_segment_Offline_TA_TO',
                        'market_segment_Online TA' : 'market_segment_Online_TA',
                        'distribution_channel_TA/TO' : 'distribution_channel_TA_TO',
                        'customer_type_Transient-Party' : 'customer_type_Transient_Party'}, inplace=True)

    y = df2["is_canceled"]
    X = df2.drop(["is_canceled"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

    print("Train data: ", X_train.shape)
    print("Test data: ", X_test.shape)
    
    ##############
    
    y_train.value_counts()
    
    ##############
    
    y_test.value_counts()
    
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
        if row['is_canceled'] == row['pred']:
            return False
        return True
    
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
    
    