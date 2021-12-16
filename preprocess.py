import pandas as pd
import numpy as np

def find_cateorical_columns(data):
    categorical_columns = []
    # data.replace({'?':np.nan},inplace=True)
    # data.dropna(axis=0,inplace=True)
    # data.reset_index(inplace=True, drop=True)
    for col in data.columns:
        levels = len(list(data[col].value_counts().index))
        if(levels < 10):
            # data[col] = data[col].astype('category')
            categorical_columns.append(col)
    return tuple(categorical_columns)


def change_dtype(data):
    # data.replace({'?':np.nan},inplace=True)
    # data.dropna(axis=0,inplace=True)
    # data.reset_index(inplace=True, drop=True)
    for col in data.columns:
        levels = len(list(data[col].value_counts().index))
        if(levels < 10):
            data[col] = data[col].astype('category')
    return data


def match_dtypes(real,synthetic):
    # synthetic = synthetic.astype(real.dtypes.to_dict())
    for col in real.columns:
            synthetic[col]=synthetic[col].astype(real[col].dtypes.name)
    # synthetic.reset_index(inplace=True, drop=True)
    return synthetic


