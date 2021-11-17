import pandas as pd
import numpy as np


def dataframe_overview(df):
    '''
    Prints the following to get a quick overview of a dataframe:
        Information returned from .info()
        First five rows (.head())
        Shape (.shape)
        List of column names
        Information returned from .describe(). The "top" is the most common value. The "freq" is the most common value’s frequency.
        Total duplicate rows

    Parameter
    ----------
    df:  pd.DataFrame
        A Pandas DataFrame

    Returns
    ----------
       None
    '''
    columns = df.columns.values.tolist()

    print("\u0332".join("INFO "))
    print(f'{df.info()}\n\n')
    print("\u0332".join("HEAD "))
    print(f'{df.head()}\n\n')
    print("\u0332".join("SHAPE "))
    print(f'{df.shape}\n\n')
    print("\u0332".join("COLUMNS "))
    print(f'{columns}\n\n')
    print("\u0332".join("COLUMN STATS "))
    print(f'{df.describe()}\n\n')
    print('\u0332'.join("TOTAL DUPLICATE ROWS "))
    print(f' {df.duplicated().sum()}')
