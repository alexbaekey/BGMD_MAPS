import pandas as pd
from . import ExtractBiasFeatures
from . import GenAtoms
from . import LearnOneRule
#from . import evaluate_ruleset

def BGMD(df, alpha, lamb1, lamb2, lamb3, delta):
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
    while cvg <= delta:
        rule = LearnOneRule.LearnOneRule(atoms, df)
        feature, condition = rule # tuple result
        ruleset = ruleset.append(rule)
        df = df - df[rule]# filter dataframe for all rows that rule captures, something like df - captured_data
        #cvg = ComputeCoverage() #?


