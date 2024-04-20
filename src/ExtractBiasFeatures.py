import pandas as pd
from scipy import stats

def ExtractBiasFeatures(df, alpha):
    '''
    input
        df :pandas dataframe: original data with "mispredict" column
        alpha :float: significance threshold
    
    internal
        A :list: list of features - delete
        I :list: misprediction indicator, flag 1 for misprediction - delete
        mg - delete
        cg - delete
        pvalue :float: results of Mann-Whitney test

    output
        bias_attr :list: the biased features
    
    '''
    bias_attr = []
    mispred = df[df['mispredict']==1]
    corpred = df[df['mispredict']==0]
    mispred = mispred.drop(['mispredict'], axis=1)
    corpred = corpred.drop(['mispredict'], axis=1)

    A = list(mispred.columns.values)
    
    for feature in A:
        res = stats.mannwhitneyu(mispred[feature],corpred[feature])
        #print(res.pvalue)
        if res.pvalue < alpha:
            bias_attr.append(feature)
    return bias_attr