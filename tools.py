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
    df = df_orig[colnames]
    a,b = colnames[:2]
    orgdf = df.groupby([a,b],sort=False,group_keys=False).mean()
    odf_reset = orgdf.reset_index()
    odf_reset.columns = colnames
    odf_pivot = odf_reset.pivot(a,b)
    pd.set_option("display.max.columns", None)
    Y = odf_pivot.columns.levels[1].values
    X = odf_pivot.index.values
    Z = odf_pivot.values.transpose()
    Xi,Yi = np.meshgrid(X, Y)
    return Xi, Yi, Z
