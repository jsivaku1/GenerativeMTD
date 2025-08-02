# mtd_utils.py
# Contains helper functions for loss calculations, statistical evaluation,
# and machine learning utility tests.

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from dython.nominal import associations

def analyze_columns(df):
    """Identifies numerical and categorical columns for the UI."""
    numerical = [{'name': col} for col in df.select_dtypes(include=np.number).columns]
    categorical = [{'name': col} for col in df.select_dtypes(exclude=np.number).columns]
    return {'numerical': numerical, 'categorical': categorical}

def mmd_rbf(X, Y, gamma=1.0):
    """Calculates the Maximum Mean Discrepancy (MMD) with an RBF kernel."""
    if X.shape[0] == 0 or Y.shape[0] == 0: return torch.tensor(0.0, device=X.device)
    XX, YY, XY = torch.cdist(X, X), torch.cdist(Y, Y), torch.cdist(X, Y)
    k_XX, k_YY, k_XY = torch.exp(-gamma * XX), torch.exp(-gamma * YY), torch.exp(-gamma * XY)
    return k_XX.mean() + k_YY.mean() - 2 * k_XY.mean()

class SinkhornDistance(torch.nn.Module):
    """Calculates the Sinkhorn divergence between two point clouds."""
    def __init__(self, eps, max_iter, device='cpu'):
        super().__init__()
        self.eps, self.max_iter, self.device = eps, max_iter, device

    def forward(self, x, y):
        if x.shape[0] == 0 or y.shape[0] == 0: return torch.tensor(0.0, device=x.device), None, None
        C = self._cost_matrix(x, y)
        x_points, y_points = x.shape[-2], y.shape[-2]
        batch_size = x.shape[0] if x.dim() > 2 else 1
        mu = torch.full((batch_size, x_points), 1.0 / x_points, dtype=torch.float, device=self.device).squeeze()
        nu = torch.full((batch_size, y_points), 1.0 / y_points, dtype=torch.float, device=self.device).squeeze()
        u, v = torch.zeros_like(mu), torch.zeros_like(nu)
        K = torch.exp(-C / self.eps)
        for _ in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            if (u - u1).abs().sum(-1).mean() < 1e-1: break
        pi = torch.exp(self.M(C, u, v))
        return torch.sum(pi * C, dim=(-2, -1)), pi, C

    def M(self, C, u, v): return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
    
    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col, y_lin = x.unsqueeze(-2), y.unsqueeze(-3)
        return torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

def dcr_nndr(real_df, synth_df):
    """Calculates Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR)."""
    real_processed = pd.get_dummies(real_df).fillna(0)
    synth_processed = pd.get_dummies(synth_df).fillna(0)
    real_processed, synth_processed = real_processed.align(synth_processed, join='inner', axis=1)
    if real_processed.empty or synth_processed.empty or len(real_processed) < 2: return np.nan, np.nan
    nn_model = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(real_processed)
    distances, _ = nn_model.kneighbors(synth_processed)
    dcr = np.mean(distances[:, 0])
    nndr = np.mean(distances[:, 0] / (distances[:, 1] + 1e-6))
    return dcr, nndr

def stat_tests(real_df, synth_df):
    """Performs statistical and privacy tests."""
    results = {}
    try:
        synth_df_aligned = synth_df[real_df.columns]
        real_corr = associations(real_df, compute_only=True)['corr']
        synth_corr = associations(synth_df_aligned, compute_only=True)['corr']
        results['pcd'] = np.linalg.norm(real_corr.fillna(0) - synth_corr.fillna(0))
        results['dcr'], results['nndr'] = dcr_nndr(real_df, synth_df_aligned)
    except Exception as e:
        print(f"Stat test error: {e}")
        results = {'pcd': np.nan, 'dcr': np.nan, 'nndr': np.nan}
    return results

def _ml_utility(train_df, test_df, target_col, model, task_type):
    """Helper function for supervised ML utility tests."""
    X_train = pd.get_dummies(train_df.drop(columns=[target_col])).fillna(0)
    y_train = train_df[target_col]
    X_test = pd.get_dummies(test_df.drop(columns=[target_col])).fillna(0)
    y_test = test_df[target_col]
    
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    if task_type == 'classification':
        probs = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, probs[:, 1]) if len(np.unique(y_test)) == 2 else roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
        return accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted'), auc
    else: # Regression
        return r2_score(y_test, preds), np.sqrt(mean_squared_error(y_test, preds)), mean_absolute_error(y_test, preds)

def predictive_model(real_df, synth_df, target_col, mode):
    """ML utility for classification tasks."""
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    if mode == 'TRTR': train_df, test_df = train_test_split(real_df, test_size=0.3, random_state=42)
    elif mode == 'TSTS': train_df, test_df = train_test_split(synth_df, test_size=0.3, random_state=42)
    elif mode == 'TSTR': train_df, test_df = synth_df, real_df
    else: train_df, test_df = real_df, synth_df # TRTS
    return _ml_utility(train_df, test_df, target_col, model, 'classification')

def regression_model(real_df, synth_df, target_col, mode):
    """ML utility for regression tasks."""
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    if mode == 'TRTR': train_df, test_df = train_test_split(real_df, test_size=0.3, random_state=42)
    elif mode == 'TSTS': train_df, test_df = train_test_split(synth_df, test_size=0.3, random_state=42)
    elif mode == 'TSTR': train_df, test_df = synth_df, real_df
    else: train_df, test_df = real_df, synth_df # TRTS
    return _ml_utility(train_df, test_df, target_col, model, 'regression')

def unsupervised_clustering_utility(real_df, synth_df):
    """Calculates clustering utility for unsupervised tasks."""
    results = {}
    try:
        real_processed = pd.get_dummies(real_df).fillna(0)
        synth_processed = pd.get_dummies(synth_df).fillna(0)
        real_processed, synth_processed = real_processed.align(synth_processed, join='inner', axis=1)

        if len(real_processed) < 2 or len(synth_processed) < 2:
            return {'Real Silhouette': np.nan, 'Synth Silhouette': np.nan, 'Real Calinski-Harabasz': np.nan, 'Synth Calinski-Harabasz': np.nan}

        n_clusters = max(2, min(8, len(real_processed) // 25))
        
        kmeans_real = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_real = kmeans_real.fit_predict(real_processed)
        if len(np.unique(labels_real)) > 1:
            results['Real Silhouette'] = silhouette_score(real_processed, labels_real)
            results['Real Calinski-Harabasz'] = calinski_harabasz_score(real_processed, labels_real)
        else:
            results['Real Silhouette'], results['Real Calinski-Harabasz'] = np.nan, np.nan

        kmeans_synth = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_synth = kmeans_synth.fit_predict(synth_processed)
        if len(np.unique(labels_synth)) > 1:
            results['Synth Silhouette'] = silhouette_score(synth_processed, labels_synth)
            results['Synth Calinski-Harabasz'] = calinski_harabasz_score(synth_processed, labels_synth)
        else:
            results['Synth Silhouette'], results['Synth Calinski-Harabasz'] = np.nan, np.nan
            
    except Exception as e:
        print(f"Could not calculate clustering utility: {e}")
        results = {'Real Silhouette': np.nan, 'Synth Silhouette': np.nan, 'Real Calinski-Harabasz': np.nan, 'Synth Calinski-Harabasz': np.nan}
        
    return results

def get_distribution_plots(real_df, synth_df, max_plots=6):
    """Prepares data for column distribution plots, showing a max of 6."""
    plot_data = {}
    
    cols_to_plot = real_df.select_dtypes(exclude=np.number).columns.tolist()
    cols_to_plot += real_df.select_dtypes(include=np.number).columns.tolist()
    cols_to_plot = cols_to_plot[:max_plots]

    for col in cols_to_plot:
        if pd.api.types.is_numeric_dtype(real_df[col].dtype):
            real_hist, bins = np.histogram(real_df[col].dropna(), bins=10)
            synth_hist, _ = np.histogram(synth_df[col].dropna(), bins=bins)
            plot_data[col] = {
                'labels': [f'{b:.2f}' for b in bins[:-1]],
                'real': real_hist.tolist(),
                'synth': synth_hist.tolist()
            }
        else:
            real_counts = real_df[col].value_counts()
            synth_counts = synth_df[col].value_counts()
            all_labels = sorted(list(set(real_counts.index) | set(synth_counts.index)))
            plot_data[col] = {
                'labels': all_labels,
                'real': [real_counts.get(l, 0) for l in all_labels],
                'synth': [synth_counts.get(l, 0) for l in all_labels]
            }
    return plot_data

def get_pca_plot_data(real_df, synth_df):
    """Performs PCA and returns data for a 2D scatter plot."""
    try:
        real_processed = pd.get_dummies(real_df).fillna(0)
        synth_processed = pd.get_dummies(synth_df).fillna(0)
        real_processed, synth_processed = real_processed.align(synth_processed, join='inner', axis=1)

        if real_processed.shape[1] < 2: return None

        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_processed)
        synth_pca = pca.transform(synth_processed)

        return {
            'real': [{'x': r[0], 'y': r[1]} for r in real_pca],
            'synth': [{'x': s[0], 'y': s[1]} for s in synth_pca]
        }
    except Exception as e:
        print(f"Could not generate PCA plot data: {e}")
        return None