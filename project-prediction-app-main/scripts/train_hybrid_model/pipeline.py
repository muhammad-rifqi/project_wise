# scripts/train_hybrid_model/pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

from ..data_preprocessor import DataLoader, DataValidator
from ..feature_engineer import FeatureTransformer, DataAggregator, DataVisualizer
from ..model_builder import GradientBoostingTrainer, NeuralNetworkTrainer, NaiveBayesTrainer, HybridFeatureFusion
from ..model_evaluator import ModelMetrics, EvaluationVisualizer, ModelComparator

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.results = None
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] TrainingPipeline: {message}")
        
    def run_data_processing(self):
        """Menjalankan fase data processing"""
        self._log("=== FASE 1: DATA PROCESSING ===")
        
        # Load data
        data_loader = DataLoader()
        data_dict = data_loader.load_all_data()
        
        if not data_dict:
            raise Exception("Gagal memuat data")
            
        # Validate data
        data_validator = DataValidator()
        data_validator.validate_data_quality(data_dict)
        data_validator.check_data_consistency(data_dict)
        
        return data_dict
        
    def run_feature_engineering(self, data_dict):
        """Menjalankan fase feature engineering"""
        self._log("=== FASE 2: FEATURE ENGINEERING ===")
        
        # Data aggregation
        aggregator = DataAggregator()
        team_agg = aggregator.aggregate_team_data(data_dict['team_members'], data_dict['allocations'])
        risk_agg = aggregator.aggregate_risk_data(data_dict['risks'])
        tech_agg = aggregator.aggregate_technology_data(data_dict['project_technologies'])
        
        # Merge datasets
        merged_data = aggregator.merge_all_datasets(
            data_dict['projects'], data_dict['allocations'], team_agg, risk_agg, tech_agg
        )
        
        # Handle missing values
        merged_data = aggregator.handle_missing_values(merged_data)
        
        # Create target variable
        transformer = FeatureTransformer()
        merged_data = transformer.create_target_variable(merged_data)
        
        # Select features and encode
        features = [
            'duration_months', 'total_development_cost', 'additional_costs',
            'total_team_size', 'avg_expertise', 'max_expertise', 'avg_salary',
            'total_risks', 'high_impact_risks', 'high_likelihood_risks',
            'unique_tech_types', 'total_tools', 'scale', 'type_project', 'sdlc_method_id'
        ]
        
        model_data = merged_data[features + ['project_success']].copy()
        model_data, categorical_features = transformer.encode_categorical_features(model_data)
        
        # Define final features
        numerical_features = [
            'duration_months', 'total_development_cost', 'additional_costs',
            'total_team_size', 'avg_expertise', 'max_expertise', 'avg_salary',
            'total_risks', 'high_impact_risks', 'high_likelihood_risks',
            'unique_tech_types', 'total_tools', 'sdlc_method_id', 'scale_encoded'
        ]
        
        X, y, final_features = transformer.select_final_features(
            model_data, numerical_features, categorical_features
        )
        
        # Visualize data
        visualizer = DataVisualizer()
        visualizer.plot_target_distribution(model_data)
        visualizer.plot_feature_distributions(model_data, numerical_features)
        visualizer.plot_correlation_matrix(model_data, numerical_features)
        
        return X, y, final_features, model_data