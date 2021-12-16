import pandas as pd
import numpy as np

def preprocess(data):
    categorical_columns = []
    data.replace({'?':np.nan},inplace=True)
    data.dropna(axis=0,inplace=True)
    data.reset_index(inplace=True, drop=True)
    for col in data.columns:
        levels = len(list(data[col].value_counts().index))
        if(levels < 10):
            data[col] = data[col].astype('category')
            categorical_columns.append(col)
    return data, categorical_columns

def match_dtypes(real,synthetic):
    for col in real.columns:
            synthetic[col]=synthetic[col].astype(synthetic[col].dtypes.name)
    synthetic.reset_index(inplace=True, drop=True)
    return synthetic


