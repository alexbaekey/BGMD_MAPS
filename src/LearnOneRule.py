import inspect

#https://www.geeksforgeeks.org/sequential-covering-algorithm/#

def LearnOneRule(atoms, df):
    '''
    input
        atoms :dict: atomic predicates, keys in dict are the feature, items are list of lambda functions
        df :pandas dataframe: the original data

    outputs
        rule :tuple: - (feature, condition)
        df_filtered :pandas df: the dataframe with data rule applies to removed
    '''
    obj = 0
    #print(df)
    for feature in atoms:
        for condition in atoms[feature]:
            indices = []
            #print(feature)
            #print(condition)
            mispred_match = 0
            mispred_nomatch = 0
            corpred_match = 0
            corpred_nomatch = 0
            #print(condition)
            for i, data in enumerate(df[feature]):
                #mask = (f(df) for f in atoms[feature])
                testbool = condition(data)
                if (testbool == True) and (df.iloc[i]['mispredict']==1):
                    mispred_match +=1
                    indices.append(i)
                    # or indices.append(df['index'].iloc[i]) ?
                elif (testbool == False) and (df.iloc[i]['mispredict']==1):
                    mispred_nomatch +=1
                elif (testbool == True) and (df.iloc[i]['mispredict']==0):
                    corpred_match +=1
                    indices.append(i)
                elif (testbool == False) and (df.iloc[i]['mispredict']==0):
                    corpred_nomatch +=1
            #TODO add printing for verbose = True in parameters file
            #print("mispred match: ", mispred_match)
            #print("mispred nomatch: ", mispred_nomatch)
            #print("corpred match: ", corpred_match)
            #print("corpred nomatch: ", corpred_nomatch)
            try:
                performance = mispred_match/(mispred_match + corpred_match)
            except:
                performance = 0
                print("divide by 0 in performance")
            try:
                recall = mispred_match/(mispred_match + mispred_nomatch)
            except:
                recall = 0
                print("divide by 0 in recall")
            new_obj = performance + recall
            if new_obj > obj:
                rule = (feature, condition) # TODO better way to store?
                obj = new_obj
                final_indices = indices
            #print("obj", obj, '\n')
            print("obj: ", obj)
    # Do I filter out all data that rule applies to, or just mispredictions rule applies to?
    return rule, final_indices
