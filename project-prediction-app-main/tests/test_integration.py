# tests/test_integration.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.datasets import make_classification  # ✅ IMPORT YANG BENAR
from scripts.data_preprocessor.loader import DataLoader
from scripts.feature_engineer.transformer import FeatureTransformer
from scripts.feature_engineer.aggregator import DataAggregator
from scripts.model_builder.gradient_boosting import GradientBoostingTrainer
from scripts.model_evaluator.metrics import ModelMetrics
from scripts.model_saver.model_serializer import ModelSerializer

class TestIntegration:
    def test_end_to_end_data_processing(self, sample_data_dict):
        """Test integrasi data processing sampai feature engineering"""
        # Data Aggregation
        aggregator = DataAggregator()
        team_agg = aggregator.aggregate_team_data(
            sample_data_dict['team_members'],
            sample_data_dict['allocations']
        )
        risk_agg = aggregator.aggregate_risk_data(sample_data_dict['risks'])
        tech_agg = aggregator.aggregate_technology_data(sample_data_dict['project_technologies'])
        
        # Data Merging
        merged_data = aggregator.merge_all_datasets(
            sample_data_dict['projects'], 
            sample_data_dict['allocations'], 
            team_agg, risk_agg, tech_agg
        )
        
        # Handle Missing Values
        merged_data = aggregator.handle_missing_values(merged_data)
        
        # Feature Engineering
        transformer = FeatureTransformer()
        merged_data = transformer.create_target_variable(merged_data)
        
        assert merged_data is not None
        assert 'project_success' in merged_data.columns
        
    def test_end_to_end_model_training(self):
        """Test integrasi model training dan evaluasi"""
        # Generate sample data
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, 
            n_redundant=2, random_state=42
        )
        
        # Model Training
        gb_trainer = GradientBoostingTrainer()
        model, selector, X_selected = gb_trainer.train_model(X, y)
        
        # Model Evaluation
        metrics_calc = ModelMetrics()
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        metrics = metrics_calc.calculate_all_metrics(y, y_pred, y_pred_proba)
        
        assert metrics is not None
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        
    def test_model_serialization_integration(self, tmp_path):
        """Test integrasi model training dan serialization"""
        # Generate sample data
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, 
            n_redundant=2, random_state=42
        )
        
        # Train model
        gb_trainer = GradientBoostingTrainer()
        model, selector, X_selected = gb_trainer.train_model(X, y)
        
        # Serialize model
        serializer = ModelSerializer(base_dir=str(tmp_path))
        model_path = serializer.save_gradient_boosting(model)
        
        assert os.path.exists(model_path)
        
        # Test bahwa model bisa digunakan setelah disimpan
        loaded_model = joblib.load(model_path)
        
        # Test prediction dengan model yang diload
        y_pred_loaded = loaded_model.predict(X)
        assert y_pred_loaded is not None
        assert len(y_pred_loaded) == len(y)