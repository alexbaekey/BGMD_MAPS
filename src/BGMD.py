import pandas as pd
from . import ExtractBiasFeatures
from . import GenAtoms

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
    return


