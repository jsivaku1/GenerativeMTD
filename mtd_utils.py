import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from dython.nominal import associations

def analyze_columns(df):
    """
    Analyzes dataframe columns and categorizes them as suitable for 
    classification or regression based on dtype and unique value counts.
    """
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        # Rule for categorical: object/category dtype, or integer with few unique values
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append({'name': col, 'type': str(df[col].dtype)})
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 20:
            categorical_cols.append({'name': col, 'type': str(df[col].dtype)})
        # Rule for numerical: float, or integer with many unique values
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append({'name': col, 'type': str(df[col].dtype)})
            
    return {'categorical': categorical_cols, 'numerical': numerical_cols}

def find_target_column(df, analysis):
    """Heuristically finds the most likely target column."""
    if analysis['categorical']:
        return analysis['categorical'][0]['name'], 'classification'
    elif analysis['numerical']:
         # A regression target is often not an ID, so we pick one with reasonable variance
        variances = df[ [c['name'] for c in analysis['numerical']] ].var()
        if not variances.empty:
            return variances.idxmax(), 'regression'
    return None, None


def PCD(real, synthetic):
    """Calculates the Pairwise Correlation Difference (PCD)."""
    if real.empty or synthetic.empty: return np.nan
    try:
        real_corr = associations(real, compute_only=True, nan_strategy='drop_samples')['corr']
        synth_corr = associations(synthetic, compute_only=True, nan_strategy='drop_samples')['corr']
    except Exception: return np.nan
    if real_corr is None or synth_corr is None: return np.nan
    return np.round(np.linalg.norm((real_corr - synth_corr), ord='fro'), 4)

def NNDR(real, synthetic):
    """Nearest Neighbor Distance Ratio."""
    real_numeric = real.select_dtypes(include=np.number)
    synthetic_numeric = synthetic.select_dtypes(include=np.number)
    if real_numeric.empty or synthetic_numeric.empty: return np.nan
    nn = NearestNeighbors(n_neighbors=2).fit(real_numeric.values)
    dists, _ = nn.kneighbors(synthetic_numeric.values)
    ratios = dists[:, 0] / (dists[:, 1] + 1e-8)
    return np.mean(ratios)

def DCR(real, synthetic):
    """Distance to Closest Record."""
    real_numeric = real.select_dtypes(include=np.number)
    synthetic_numeric = synthetic.select_dtypes(include=np.number)
    if real_numeric.empty or synthetic_numeric.empty: return np.nan
    nn = NearestNeighbors(n_neighbors=1).fit(real_numeric.values)
    dists, _ = nn.kneighbors(synthetic_numeric.values)
    return np.mean(dists)

def stat_tests(real, synthetic):
    """Aggregate main stat tests into a dict."""
    return {'pcd': PCD(real, synthetic), 'nndr': NNDR(real, synthetic), 'dcr': DCR(real, synthetic)}

def predictive_model(real, synthetic, class_col, mode='TSTR'):
    """Classification evaluation across TSTR, TRTS, TRTR, TSTS."""
    if class_col not in real.columns or class_col not in synthetic.columns: return 0.0, 0.0
    X_real = pd.get_dummies(real.drop(class_col, axis=1))
    y_real = real[class_col]
    X_synth = pd.get_dummies(synthetic.drop(class_col, axis=1))
    y_synth = synthetic[class_col]
    
    common_cols = X_real.columns.intersection(X_synth.columns)
    X_real, X_synth = X_real[common_cols], X_synth[common_cols]

    if mode == 'TSTR': X_train, y_train, X_test, y_test = X_synth, y_synth, X_real, y_real
    elif mode == 'TRTS': X_train, y_train, X_test, y_test = X_real, y_real, X_synth, y_synth
    elif mode == 'TRTR': X_train, y_train, X_test, y_test = X_real, y_real, X_real, y_real
    elif mode == 'TSTS': X_train, y_train, X_test, y_test = X_synth, y_synth, X_synth, y_synth
    else: raise ValueError("Mode must be one of TSTR, TRTS, TRTR, TSTS")

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions), f1_score(y_test, predictions, average='weighted')

def regression_model(real, synthetic, target_col, mode='TSTR'):
    """Regression evaluation across TSTR, TRTS, TRTR, TSTS."""
    if target_col not in real.columns or target_col not in synthetic.columns: return 0.0, 0.0, 0.0
    X_real = pd.get_dummies(real.drop(target_col, axis=1))
    y_real = real[target_col]
    X_synth = pd.get_dummies(synthetic.drop(target_col, axis=1))
    y_synth = synthetic[target_col]
    
    common_cols = X_real.columns.intersection(X_synth.columns)
    X_real, X_synth = X_real[common_cols], X_synth[common_cols]

    if mode == 'TSTR': X_train, y_train, X_test, y_test = X_synth, y_synth, X_real, y_real
    elif mode == 'TRTS': X_train, y_train, X_test, y_test = X_real, y_real, X_synth, y_synth
    else: raise ValueError("Mode for regression must be TSTR or TRTS")

    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return r2, mae, rmse
