# scripts/model_builder/gradient_boosting.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from datetime import datetime

class GradientBoostingTrainer:
    def __init__(self):
        self.model = None
        self.selector = None
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] GradientBoosting: {message}")
        
    def train_model(self, X_train, y_train):
        """Training Gradient Boosting model dan return model + selector"""
        self._log("Memulai training Gradient Boosting")
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Feature selection
        self._log("Melakukan feature selection")
        self.selector = SelectFromModel(self.model, prefit=True, threshold='median')
        X_train_selected = self.selector.transform(X_train)
        
        selected_features_mask = self.selector.get_support()
        
        self._log(f"Feature selection selesai: {X_train_selected.shape[1]} features terpilih")
        
        # Return model DAN selector
        return self.model, self.selector, X_train_selected
        
    def perform_feature_selection(self, X_train):
        """Feature selection menggunakan Gradient Boosting"""
        self._log("Melakukan feature selection")
        
        self.selector = SelectFromModel(self.model, prefit=True, threshold='median')
        X_train_selected = self.selector.transform(X_train)
        
        selected_features_mask = self.selector.get_support()
        
        self._log(f"Feature selection selesai: {X_train_selected.shape[1]} features terpilih")
        
        return X_train_selected, selected_features_mask
        
    def get_feature_importance(self, feature_names):
        """Mendapatkan feature importance"""
        self._log("Analisis feature importance")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self._log("Top 5 Feature Importance:")
        for i, row in feature_importance.head().iterrows():
            self._log(f"  - {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance