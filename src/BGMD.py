import pandas as pd

def BGMD(df):
    '''
    implementation of Bias Guided Misprediction Diagnoser
    input: df
    output: ruleset
    '''
    # collect features as list
    features = list(df.columns.values)



