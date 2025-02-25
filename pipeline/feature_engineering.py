# pipeline/feature_engineering.py

import os
import numpy as np
import pandas as pd
import featuretools as ft
from pipeline.data_loader import load_data, get_absolute_path

class FeatureEngineer:
    def __init__(self, file_path: str, file_type: str = None, table_name: str = None):
        """
        Loads data and ensures an 'id' column for Featuretools.
        """
        abs_file_path = get_absolute_path(file_path)
        self.df = load_data(abs_file_path, file_type, table_name)
        # Make sure we have a unique ID column for Featuretools
        if "id" not in self.df.columns:
            self.df.reset_index(drop=False, inplace=True)
            self.df.rename(columns={'index': 'id'}, inplace=True)

    def create_additional_features(self):
        """
        Simple demonstration of new features. 
        We skip 'id' from numeric transformations.
        """
        numeric_cols = [c for c in self.df.select_dtypes(include=['float64', 'int64']).columns if c != 'id']

        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            # ratio
            self.df[f'ratio_{col1}_{col2}'] = self.df[col1] / (self.df[col2] + 1e-9)
            # interaction
            self.df[f'interaction_{col1}_{col2}'] = self.df[col1] * self.df[col2]
            # difference
            self.df[f'diff_{col1}_{col2}'] = self.df[col1] - self.df[col2]

        if len(numeric_cols) >= 1:
            col1 = numeric_cols[0]
            # squared
            self.df[f'squared_{col1}'] = self.df[col1] ** 2
            # binned
            self.df[f'binned_{col1}'] = pd.qcut(self.df[col1], q=4, labels=False, duplicates='drop')

        # sum & mean across numeric
        if len(numeric_cols) > 0:
            self.df['sum_numeric'] = self.df[numeric_cols].sum(axis=1)
            self.df['mean_numeric'] = self.df[numeric_cols].mean(axis=1)

        # an example ratio if we have 3 or more numeric
        if len(numeric_cols) >= 3:
            col3 = numeric_cols[2]
            self.df[f'ratio_{col2}_{col3}'] = self.df[col2] / (self.df[col3] + 1e-9)

        # log of first numeric col if not already log-transformed
        if len(numeric_cols) >= 1:
            col1 = numeric_cols[0]
            # only apply log if > 0
            self.df[f'log_{col1}'] = self.df[col1].apply(lambda x: np.log1p(x) if x > 0 else x)

        # count zero or negative across numeric
        if len(numeric_cols) > 0:
            self.df['cnt_zero_neg'] = (self.df[numeric_cols] <= 0).sum(axis=1)

    def generate_features(self):
        """
        Uses Featuretools DFS to auto-generate additional features.
        """
        es = ft.EntitySet(id="data")
        es = es.add_dataframe(
            dataframe_name="main",
            dataframe=self.df,
            index="id"
        )
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="main",
            max_depth=2,
            verbose=True
        )
        self.feature_matrix = feature_matrix
        return feature_matrix

    def save_engineered_data(self, output_filename="engineered_data.csv"):
        processed_dir = get_absolute_path("data/processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        output_path = os.path.join(processed_dir, output_filename)
        self.feature_matrix.to_csv(output_path, index=False)
        print(f"Engineered data saved to: {output_path}")

    def run(self):
        """
        Combined method: create custom features + run DFS.
        """
        self.create_additional_features()
        self.generate_features()

