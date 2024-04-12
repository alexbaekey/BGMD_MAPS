import pandas as pd

def mispredict_label(df, truth, pred):
    '''
    input
        df :pandas dataframe:
        model_output :: .... #TODO
        truth :string: column name of data labels
        pred :string: column name of predicted value, model output

    output
        df_flagged :pandas dataframe: modified df with "mispredict" column
    '''
    df_flagged = df.copy().drop([truth, pred], axis=1)

    for index, row in df.iterrows():
        if row[truth] == row[pred]:
            df_flagged.loc[index, 'mispredict'] = False
        else:
            df_flagged.loc[index, 'mispredict'] = True
    return df_flagged
