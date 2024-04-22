import pandas as pd
from . import ExtractBiasFeatures
from . import GenAtoms
from . import LearnOneRule
#from . import evaluate_ruleset

def BGMD(df, alpha, lamb1, lamb2, lamb3, delta):
    df = df.reset_index()
    '''
    implementation of Bias Guided Misprediction Diagnoser
    
    input: 
        df :pandas dataframe: - original data 
        alpha :int: - Mann Whitney U test significance threshold for ExtractBiasedFeatures
        lamb1 :int: - lambda_1 in objective equation, scaling for precision
        lamb2 :int: - lambda_1 in objective equation, scaling for recall
        lamb1 :int: - lambda_1 in objective equation, scaling for rule size
        delta :int: - threshold coverage for rule learning

    internal:
        BA :list: list of biased features

    output: 
        ruleset
    '''
    print("All Attributes:")
    print(df.columns)
    print('\n\n')
    BA = ExtractBiasFeatures.ExtractBiasFeatures(df, alpha)
    print("Biased Attributes:")
    print(BA) 
    print('\n\n')
    atoms = GenAtoms.GenAtoms(df, BA)
    
    # Rule learning - sequential covering algo
    # https://www.geeksforgeeks.org/sequential-covering-algorithm/#
    ruleset = []
    cvg = 0
    orig_df = df.copy()
    while cvg <= delta:
        rule, indices = LearnOneRule.LearnOneRule(atoms, df)
        print(len(indices)) #TODO chosen rule is taking 14835/14836 data rows....
        feature, condition = rule # tuple result
        ruleset.append(rule)
        print(ruleset)
        df_rules = df.index.isin(indices)# filter dataframe for all rows that rule captures, something like df - captured_data
        #df_new = df[~df_rules] # just sets index in df to false, doesnt remove
        # ^ this result contradicts this SO answer: https://stackoverflow.com/questions/28256761/select-pandas-rows-by-excluding-index-number
        print("len of origin df:", len(orig_df))
        df_new = df.drop(indices, axis=0)
        print("len of new df:", len(df_new))
        cvg = len(df_new)/len(orig_df) # guessing this is just data rules apply to over original dataset
        cvg = len(df_rules)/len(orig_df) # guessing this is just data rules apply to over original dataset
        df = df_new.copy()
        # Take out learned rule
        #del(atoms[feature][rule] or something like that
        #TODO cvg is miscalculated, returns 0.999ish every iterations
        print("cvg:", cvg)


