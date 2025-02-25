# pipeline/data_cleaning.py

import pandas as pd
import numpy as np
import os
from pipeline.data_loader import load_data, get_absolute_path
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self, file_path: str, file_type: str = None, table_name: str = None):
        """
        Initializes the DataCleaner with a loaded DataFrame.
        """
        abs_file_path = get_absolute_path(file_path)
        self.df = load_data(abs_file_path, file_type, table_name)

    def handle_missing_values(self):
        """
        Fill missing values:
         - Numeric columns => median
         - Categorical => mode
        """
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                # Instead of inplace=, do direct assignment to avoid FutureWarning
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
            else:
                # Categorical
                mode_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else ''
                self.df[col] = self.df[col].fillna(mode_val)

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

    def transform_features(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            # Log transform if strictly positive and skew>1
            if (self.df[col] > 0).all() and (self.df[col].skew() > 1):
                self.df[col] = np.log1p(self.df[col])

    def encode_categorical(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

    def apply_scaling(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        # Exclude binary columns (i.e., columns with exactly 2 unique values)
        cols_to_scale = [col for col in numeric_cols if self.df[col].nunique() > 2]
        if cols_to_scale:
            scaler = StandardScaler()
            self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])

    def clean_data(self):
        self.handle_missing_values()
        self.remove_duplicates()
        self.transform_features()
        self.encode_categorical()
        self.apply_scaling()
        return self.df

    def save_cleaned_data(self, output_filename="cleaned_data.csv"):
        processed_dir = get_absolute_path("data/processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        output_path = os.path.join(processed_dir, output_filename)
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

