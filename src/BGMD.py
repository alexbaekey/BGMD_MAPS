import pandas as pd
from . import ExtractBiasFeatures

def BGMD(df, alpha):
    '''
    implementation of Bias Guided Misprediction Diagnoser
    input: df
    output: ruleset
    '''
    return ExtractBiasFeatures.ExtractBiasFeatures(df, alpha)


