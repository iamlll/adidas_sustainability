import pandas as pd
import numpy as np

def parse_CSV(df_orig,colnames):
    '''
    inputs:
        filename: file from which to read into Pandas dataframe
        colnames: array of names of 3 columns (x,y,z) to plot
    outputs:
        arrays of x,y,z formatted for contour plotting
    '''
    #read in CSV as Pandas dataframe
    import numpy.ma as ma
    a,b,c = colnames
    Z = df_orig.pivot_table(index=a, columns=b, values=c).T.values
    X_unique = np.sort(df_orig[a].unique())
    Y_unique = np.sort(df_orig[b].unique())
    Xi, Yi = np.meshgrid(X_unique, Y_unique)
    return Xi, Yi, Z
