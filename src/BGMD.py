import pandas as pd
from . import ExtractBiasFeatures
from . import GenAtoms
#from . import evaluate_ruleset

def BGMD(df, alpha):
    '''
    implementation of Bias Guided Misprediction Diagnoser
    input: df
    internal:
        BA :list: list of biased features
    output: ruleset
    '''

    BA = ExtractBiasFeatures.ExtractBiasFeatures(df, alpha)
    atoms = GenAtoms.GenAtoms(df, BA)
    print(atoms) 
    # Rule learning
    '''
    ruleset = []

    for index, data in df.iterrows():
        for feature in BA:
            #print(data[feature])
            #print(atoms[feature].v)
            if atoms[feature](data[feature]):
                continue
            else:
                break #TODO, make sure this breaks correctly
        ruleset.append(atoms[feature])
    '''
    #https://www.geeksforgeeks.org/sequential-covering-algorithm/#
    cvg = 0
    while cvg <= delta:
        ruleset = ...
        phi = LearnRule()
        #precision, recall, rulesize = evaluate_ruleset
    #return ruleset


