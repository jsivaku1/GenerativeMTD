# data_pipeline.py
# A robust data preprocessing pipeline to handle mixed data types, imputation,
# scaling, and one-hot encoding, with a corrected inverse_transform method.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPipeline:
    """
    A robust data processing pipeline that correctly handles fitting,
    transformation, and inverse-transformation for mixed data types.
    """
    def __init__(self):
        self._is_fitted = False
        self.preprocessor = None
        self.original_dtypes = None
        self.original_columns = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.cat_lengths = []
        self.output_dim = 0

    def fit(self, df):
        """
        Learns the data structure and fits the entire preprocessing pipeline.
        This method identifies column types and sets up the correct transformers.
        """
        df.columns = df.columns.astype(str)
        self.original_dtypes = df.dtypes
        self.original_columns = df.columns.tolist()
        
        for col in df.select_dtypes(include=['object']).columns:
            if not pd.to_numeric(df[col], errors='coerce').notna().all():
                 df[col] = df[col].astype(str)

        self.numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        transformers = []
        
        if self.numerical_cols:
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, self.numerical_cols))

        if self.categorical_cols:
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_cols))
        
        if not transformers: 
            raise ValueError("DataFrame must have at least one numerical or categorical column.")

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        self.preprocessor.fit(df)
        
        self.output_dim = len(self.numerical_cols)
        if self.categorical_cols:
            ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            self.cat_lengths = [len(cats) for cats in ohe.categories_]
            self.output_dim += sum(self.cat_lengths)

        self._is_fitted = True
        
    def transform(self, df):
        """Transforms raw data using the fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming data.")
        df.columns = df.columns.astype(str)
        return self.preprocessor.transform(df)

    def inverse_transform(self, data):
        """
        Correctly brings transformed data back to its original format.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before inverse transforming data.")
        
        inversed_dfs = []
        current_col_idx = 0

        if self.numerical_cols:
            num_pipeline = self.preprocessor.named_transformers_['num']
            scaler = num_pipeline.named_steps['scaler']
            num_feature_count = len(self.numerical_cols)
            num_data = data[:, current_col_idx : current_col_idx + num_feature_count]
            inversed_num = pd.DataFrame(scaler.inverse_transform(num_data), columns=self.numerical_cols)
            inversed_dfs.append(inversed_num)
            current_col_idx += num_feature_count
        
        if self.categorical_cols:
            cat_pipeline = self.preprocessor.named_transformers_['cat']
            onehot = cat_pipeline.named_steps['onehot']
            cat_data = data[:, current_col_idx:]
            inversed_cat = pd.DataFrame(onehot.inverse_transform(cat_data), columns=self.categorical_cols)
            inversed_dfs.append(inversed_cat)

        if not inversed_dfs:
            return pd.DataFrame(columns=self.original_columns)

        df_final = pd.concat(inversed_dfs, axis=1)
        
        df_final = df_final[self.original_columns]
        
        for col, dtype in self.original_dtypes.items():
            try:
                if pd.api.types.is_numeric_dtype(dtype):
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                    if pd.api.types.is_integer_dtype(dtype):
                        df_final[col] = df_final[col].astype('Int64').round()
                df_final[col] = df_final[col].astype(dtype, errors='ignore')
            except Exception:
                pass
                
        return df_final