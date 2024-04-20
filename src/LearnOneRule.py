import inspect

#https://www.geeksforgeeks.org/sequential-covering-algorithm/#

def LearnOneRule(atoms, df):
    for feature in atoms:
        #print(feature)
        #print(atoms[feature])
        for condition in atoms[feature]:
            print(inspect.getsource(condition))
            match = 0
            nomatch = 0
            #print(condition)
            #TODO split up data into mispredicted/correctly predicted
            for data in df[feature]:
                #mask = (f(df) for f in atoms[feature])
                testbool = condition(data)
                if testbool == True:
                    match +=1
                elif testbool == False:
                    nomatch +=1
            print(feature)
            print(condition)
            print("matches: ", match)
            print("nomatch: ", nomatch) 
    
    #new_rule = atom
    #covered = # data that new rule includes
    #df = df-covered
    #rules = rules+new_rule
    #return df, rules
