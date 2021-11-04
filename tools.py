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
    df = df_orig[colnames]
    a,b,c = colnames
    orgdf = df.groupby([a,b]).mean() #group by eta and U values; take the mean since there's only one of each combo anyways so it doesn't matter
    odf_reset = orgdf.reset_index()
    odf_reset.columns = colnames
    odf_pivot = odf_reset.pivot(a,b)
    pd.set_option("display.max.columns", None)
    Y = odf_pivot.columns.levels[1].values
    X = odf_pivot.index.values
    Z = odf_pivot.values.transpose()
    Xi,Yi = np.meshgrid(X, Y)
    return Xi, Yi, Z
