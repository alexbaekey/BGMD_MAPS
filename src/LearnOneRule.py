import inspect

#https://www.geeksforgeeks.org/sequential-covering-algorithm/#

def LearnOneRule(atoms, df):
    obj = 0
    print(df.head())
    for feature in atoms:
        #print(feature)
        #print(atoms[feature])
        for condition in atoms[feature]:
            print(feature)
            print(condition)
            #print(inspect.getsource(condition))
            mispred_match = 0
            mispred_nomatch = 0
            corpred_match = 0
            corpred_nomatch = 0
            #print(condition)
            #TODO split up data into mispredicted/correctly predicted
            for i, data in enumerate(df[feature]):
                #mask = (f(df) for f in atoms[feature])
                testbool = condition(data)
                if (testbool == True) and (df.iloc[i]['mispredict']==1):
                    mispred_match +=1
                elif (testbool == False) and (df.iloc[i]['mispredict']==1):
                    mispred_nomatch +=1
                elif (testbool == True) and (df.iloc[i]['mispredict']==0):
                    corpred_match +=1
                elif (testbool == False) and (df.iloc[i]['mispredict']==0):
                    corpred_nomatch +=1
            print("mispred match: ", mispred_match)
            print("mispred nomatch: ", mispred_nomatch)
            print("corpred match: ", corpred_match)
            print("corpred nomatch: ", corpred_nomatch)
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
            print("obj", obj)
            if new_obj > obj:
                rule = (feature, condition) # TODO better way to store?
                obj = new_obj
    #TODO remove rule from ruleset, remove applicable data from dataset
    print(rule)
    return rule
