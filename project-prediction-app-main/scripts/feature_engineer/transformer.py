# scripts/feature_engineer/transformer.py
import pandas as pd
import numpy as np
from datetime import datetime

class FeatureTransformer:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] FeatureTransformer: {message}")
        
    def encode_categorical_features(self, model_data):
        """Encoding categorical variables"""
        self._log("Memulai encoding categorical features")
        
        # Label encoding untuk scale
        scale_mapping = {'low': 0, 'medium': 1, 'high': 2}
        model_data['scale_encoded'] = model_data['scale'].map(scale_mapping)

        # One-hot encoding untuk type_project
        type_project_dummies = pd.get_dummies(model_data['type_project'], prefix='type')
        model_data = pd.concat([model_data, type_project_dummies], axis=1)
        
        self._log("Encoding categorical features selesai")
        
        return model_data, type_project_dummies.columns.tolist()
        
    def create_target_variable(self, merged_data):
        """Membuat target variable untuk modeling"""
        self._log("Membuat target variable")
        
        merged_data['project_success'] = (merged_data['status_project'] == 'success').astype(int)
        
        success_rate = merged_data['project_success'].mean()
        self._log(f"Success rate: {success_rate:.2%}")
        
        return merged_data
        
    def select_final_features(self, model_data, numerical_features, categorical_features):
        """Memilih final features untuk modeling"""
        self._log("Memilih final features")
        
        final_features = numerical_features + categorical_features
        X = model_data[final_features]
        y = model_data['project_success']
        
        self._log(f"Final features: {len(final_features)}")
        self._log(f"Numerical features: {len(numerical_features)}")
        self._log(f"Categorical features: {len(categorical_features)}")
        
        return X, y, final_features