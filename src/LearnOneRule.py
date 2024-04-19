import inspect

#https://www.geeksforgeeks.org/sequential-covering-algorithm/#

def LearnOneRule(atoms, df):
    for feature in atoms:
        for condition in atoms[feature]:
            print(inspect.getsource(condition))
            match = 0
            nomatch = 0
            for data in df[feature]:
                print(type(data))
                print(type(condition))
                print(len(condition))
                print(inspect.getsource(condition))
                #print(data)
                testbool = condition(data)
                print(type(testbool))
                if testbool == True:
                    match +=1
                elif testbool == False:
                    nomatch +=1
        print("matches: ", match)
        print("nomatch: ", nomatch) 



    #new_rule = atom
    #covered = # data that new rule includes
    #df = df-covered
    #rules = rules+new_rule
    #return df, rules
