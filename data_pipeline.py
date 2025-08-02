import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPipeline:
    """
    A developer-grade data processing pipeline for cleaning, transforming,
    and inverse-transforming tabular data for machine learning models.
    
    This version correctly implements a stateful fit/transform process and a
    robust inverse_transform method to prevent common sklearn errors, especially
    for datasets with only numerical or only categorical columns.
    """
    def __init__(self):
        self.numerical_cols = []
        self.categorical_cols = []
        self.preprocessor = None
        self.original_dtypes = None
        self.original_columns = None
        self._is_fitted = False

    def fit(self, df):
        """
        Learns the data structure and fits the entire preprocessing pipeline.
        """
        self.original_dtypes = df.dtypes
        self.original_columns = df.columns.tolist()

        # Identify column types from the original dataframe
        self.numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        transformers = []
        
        # Add numeric pipeline only if there are numeric columns
        if self.numerical_cols:
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, self.numerical_cols))
        
        # Add categorical pipeline only if there are categorical columns
        if self.categorical_cols:
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_cols))
        
        if not transformers:
            raise ValueError("Dataframe must contain at least one numerical or categorical column.")

        # Create the main ColumnTransformer preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        self.preprocessor.fit(df)
        self._is_fitted = True
        
    def transform(self, df):
        """
        Transforms raw data using the fitted pipeline.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call fit() first.")
        return self.preprocessor.transform(df)

    def inverse_transform(self, data_transformed):
        """
        Converts transformed data back to its original format and data types.
        This method correctly handles the inverse transformation by accessing
        the individual transformers within the ColumnTransformer and checking
        if numerical or categorical columns exist.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call inverse_transform() first.")
        
        inversed_dfs = []
        current_col_idx = 0

        # Handle numerical columns if they exist
        if self.numerical_cols:
            num_pipeline = self.preprocessor.named_transformers_['num']
            num_scaler = num_pipeline.named_steps['scaler']
            num_feature_count = len(self.numerical_cols)
            
            num_data = data_transformed[:, current_col_idx : current_col_idx + num_feature_count]
            inversed_num = pd.DataFrame(num_scaler.inverse_transform(num_data), columns=self.numerical_cols)
            inversed_dfs.append(inversed_num)
            current_col_idx += num_feature_count
        
        # Handle categorical columns if they exist
        if self.categorical_cols:
            cat_pipeline = self.preprocessor.named_transformers_['cat']
            cat_onehot = cat_pipeline.named_steps['onehot']
            
            cat_data = data_transformed[:, current_col_idx:]
            inversed_cat = pd.DataFrame(cat_onehot.inverse_transform(cat_data), columns=self.categorical_cols)
            inversed_dfs.append(inversed_cat)

        if not inversed_dfs:
            return pd.DataFrame(columns=self.original_columns)

        # Combine the inversed dataframes
        df_final = pd.concat(inversed_dfs, axis=1)
        
        # Reorder columns to match the original dataframe exactly
        df_final = df_final[self.original_columns]
        
        # Restore original data types, handling potential conversion errors
        for col in df_final.columns:
            original_type = self.original_dtypes[col]
            try:
                if pd.api.types.is_numeric_dtype(original_type):
                    if np.issubdtype(original_type, np.integer):
                        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').round().astype(original_type)
                    else:
                        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').astype(original_type)
                else:
                    df_final[col] = df_final[col].astype(original_type)
            except (ValueError, TypeError):
                pass
                
        return df_final
