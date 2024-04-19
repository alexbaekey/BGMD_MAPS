#TODO categorical variables
import pandas as pd
import numpy as np
import inspect

def GenAtoms(df, BA):
    '''
    df :pandas dataframe: mispredicted data
    BA :list: biased features
    '''
    atomic_predicates = {}
    for feature in BA:
        x = df[feature]
        x_sorted = x.sort_values()
        
        #binning 
        #use of divmod to split into 4 roughly equal sized bins
        #https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
        #k,m = lambda x: divmod(len(x), 4)
        #OR just np.array_split, seems simpler
        a,b,c,d = np.array_split(x_sorted, 4)
        
        #create predicates x op c where x is feature, op is <= or > and c is max of each bin
        #c1, c2, c3, c4 = max(a), max(b), max(c), max(d)
        constants = [a,b,c,d]
        atomic_predicates[feature] = [  lambda x: x < a, \
                                        lambda x: x >= a, \
                                        lambda x: x < b, \
                                        lambda x: x >= b, \
                                        lambda x: x < c, \
                                        lambda x: x >= c, \
                                        lambda x: x < d, \
                                        lambda x: x >= d ]
        
        #print(atomic_predicates[feature])

    #for key, value in atomic_predicates.items():
        #print(key, value)
        #print(inspect.getsource(value))
    return atomic_predicates
