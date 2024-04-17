#https://www.geeksforgeeks.org/sequential-covering-algorithm/#
def LearnRule(atom, df):
    new_rule = atom
    covered = # data that new rule includes
    df = df-covered
    rules = rules+new_rule
    return df, rules
