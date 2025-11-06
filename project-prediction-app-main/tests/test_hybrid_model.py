# tests/test_hybrid_model.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
from scripts.train_hybrid_model.trainer import HybridModelTrainer, TrainingPipeline
from scripts.train_hybrid_model.config import ModelConfig
from scripts.feature_engineer.aggregator import DataAggregator  # ✅ TAMBAHKAN IMPORT INI
from scripts.feature_engineer.transformer import FeatureTransformer  # ✅ TAMBAHKAN IMPORT INI

class TestModelConfig:
    def test_config_initialization(self):
        """Test inisialisasi ModelConfig"""
        config = ModelConfig()
        assert config is not None
        assert config.RANDOM_STATE == 42
        assert config.TEST_SIZE == 0.2
        assert config.MODEL_WEIGHTS is not None
        
    def test_config_custom_values(self):
        """Test ModelConfig dengan nilai custom"""
        config = ModelConfig()
        
        # Test bahwa nilai default ada
        assert hasattr(config, 'RANDOM_STATE')
        assert hasattr(config, 'TEST_SIZE')
        assert hasattr(config, 'MODEL_WEIGHTS')
        
        # Test bahwa MODEL_WEIGHTS adalah dictionary dengan keys yang benar
        assert 'gradient_boosting' in config.MODEL_WEIGHTS
        assert 'neural_network' in config.MODEL_WEIGHTS
        assert 'naive_bayes' in config.MODEL_WEIGHTS

class TestTrainingPipeline:
    def test_pipeline_initialization(self):
        """Test inisialisasi TrainingPipeline"""
        config = ModelConfig()
        pipeline = TrainingPipeline(config)
        assert pipeline is not None
        assert pipeline.config == config
        
    def test_run_data_processing(self, mocker):
        """Test fase data processing"""
        config = ModelConfig()
        pipeline = TrainingPipeline(config)
        
        # Mock data loading untuk avoid file dependencies
        mock_loader = mocker.Mock()
        mock_loader.load_all_data.return_value = {
            'projects': pd.DataFrame({'dummy': [1, 2, 3]}),
            'allocations': pd.DataFrame({'dummy': [1, 2, 3]}),
            'team_members': pd.DataFrame({'dummy': [1, 2, 3]}),
            'risks': pd.DataFrame({'dummy': [1, 2, 3]}),
            'project_technologies': pd.DataFrame({'dummy': [1, 2, 3]})
        }
        
        mocker.patch('scripts.data_preprocessor.loader.DataLoader', return_value=mock_loader)
        mocker.patch('scripts.data_preprocessor.validator.DataValidator')
        
        data_dict = pipeline.run_data_processing()
        
        assert data_dict is not None
        assert isinstance(data_dict, dict)
        
    def test_run_feature_engineering(self, mocker):
        """Test fase feature engineering"""
        config = ModelConfig()
        pipeline = TrainingPipeline(config)
        
        # Sample data dictionary
        sample_data_dict = {
            'projects': pd.DataFrame({
                'id': [1, 2, 3],
                'project_name': ['Project A', 'Project B', 'Project C'],
                'duration_months': [6, 12, 8],
                'total_development_cost': [50000, 100000, 75000],
                'additional_costs': [5000, 10000, 7500],
                'scale': ['medium', 'high', 'low'],
                'type_project': ['web', 'mobile', 'desktop'],
                'sdlc_method_id': [1, 2, 1],
                'status_project': ['success', 'failed', 'success'],
                'start_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
                'end_date': ['2023-07-01', '2024-02-01', '2023-11-01']
            }),
            'allocations': pd.DataFrame({
                'id': [1, 2, 3],
                'project_id': [1, 2, 3]
            }),
            'team_members': pd.DataFrame({
                'allocation_id': [1, 1, 2, 2, 3],
                'quantity': [2, 3, 1, 2, 4],
                'expertise_level_id': [5, 6, 4, 5, 7],
                'avg_salary': [5000, 6000, 4500, 5500, 7000]
            }),
            'risks': pd.DataFrame({
                'project_id': [1, 1, 2, 3],
                'impact_level': ['high', 'medium', 'low', 'high'],
                'likelihood': ['medium', 'high', 'low', 'medium']
            }),
            'project_technologies': pd.DataFrame({
                'project_id': [1, 1, 2, 3],
                'technology_type_id': [1, 2, 1, 3],
                'tool_name': ['React', 'Node.js', 'Flutter', 'Django']
            })
        }
        
        # Mock DataAggregator methods
        mocker.patch.object(DataAggregator, 'aggregate_team_data', return_value=pd.DataFrame({
            'project_id': [1, 2, 3],
            'total_team_size': [5, 3, 4],
            'avg_expertise': [5.5, 4.0, 7.0],
            'max_expertise': [6, 4, 7],
            'avg_salary': [5500, 4500, 7000]
        }))
        
        mocker.patch.object(DataAggregator, 'aggregate_risk_data', return_value=pd.DataFrame({
            'project_id': [1, 2, 3],
            'total_risks': [2, 1, 1],
            'high_impact_risks': [1, 0, 0],
            'high_likelihood_risks': [1, 0, 1]
        }))
        
        mocker.patch.object(DataAggregator, 'aggregate_technology_data', return_value=pd.DataFrame({
            'project_id': [1, 2, 3],
            'unique_tech_types': [2, 1, 1],
            'total_tools': [2, 1, 1]
        }))
        
        mocker.patch.object(DataAggregator, 'merge_all_datasets', return_value=pd.DataFrame({
            'id': [1, 2, 3],
            'project_name': ['Project A', 'Project B', 'Project C'],
            'duration_months': [6, 12, 8],
            'total_development_cost': [50000, 100000, 75000],
            'additional_costs': [5000, 10000, 7500],
            'scale': ['medium', 'high', 'low'],
            'type_project': ['web', 'mobile', 'desktop'],
            'sdlc_method_id': [1, 2, 1],
            'status_project': ['success', 'failed', 'success'],
            'total_team_size': [5, 3, 4],
            'avg_expertise': [5.5, 4.0, 7.0],
            'max_expertise': [6, 4, 7],
            'avg_salary': [5500, 4500, 7000],
            'total_risks': [2, 1, 1],
            'high_impact_risks': [1, 0, 0],
            'high_likelihood_risks': [1, 0, 1],
            'unique_tech_types': [2, 1, 1],
            'total_tools': [2, 1, 1]
        }))
        
        mocker.patch.object(DataAggregator, 'handle_missing_values', side_effect=lambda x: x)
        
        # Mock FeatureTransformer methods
        mocker.patch.object(FeatureTransformer, 'create_target_variable', side_effect=lambda x: x.assign(project_success=[1, 0, 1]))
        mocker.patch.object(FeatureTransformer, 'encode_categorical_features', return_value=(
            pd.DataFrame({
                'duration_months': [6, 12, 8],
                'total_development_cost': [50000, 100000, 75000],
                'additional_costs': [5000, 10000, 7500],
                'total_team_size': [5, 3, 4],
                'avg_expertise': [5.5, 4.0, 7.0],
                'max_expertise': [6, 4, 7],
                'avg_salary': [5500, 4500, 7000],
                'total_risks': [2, 1, 1],
                'high_impact_risks': [1, 0, 0],
                'high_likelihood_risks': [1, 0, 1],
                'unique_tech_types': [2, 1, 1],
                'total_tools': [2, 1, 1],
                'sdlc_method_id': [1, 2, 1],
                'scale_encoded': [1, 2, 0],
                'type_web': [1, 0, 0],
                'type_mobile': [0, 1, 0],
                'type_desktop': [0, 0, 1],
                'project_success': [1, 0, 1]
            }),
            ['type_web', 'type_mobile', 'type_desktop']
        ))
        
        mocker.patch.object(FeatureTransformer, 'select_final_features', return_value=(
            pd.DataFrame({
                'duration_months': [6, 12, 8],
                'total_development_cost': [50000, 100000, 75000],
                # ... other features
            }),
            pd.Series([1, 0, 1]),
            ['duration_months', 'total_development_cost', 'additional_costs', 'total_team_size', 
             'avg_expertise', 'max_expertise', 'avg_salary', 'total_risks', 'high_impact_risks', 
             'high_likelihood_risks', 'unique_tech_types', 'total_tools', 'sdlc_method_id', 
             'scale_encoded', 'type_web', 'type_mobile', 'type_desktop']
        ))
        
        # Mock visualizer methods
        mocker.patch('scripts.feature_engineer.visualizer.DataVisualizer.plot_target_distribution')
        mocker.patch('scripts.feature_engineer.visualizer.DataVisualizer.plot_feature_distributions')
        mocker.patch('scripts.feature_engineer.visualizer.DataVisualizer.plot_correlation_matrix')
        
        X, y, final_features, model_data = pipeline.run_feature_engineering(sample_data_dict)
        
        assert X is not None
        assert y is not None
        assert final_features is not None
        assert model_data is not None
        assert len(final_features) > 0

class TestHybridModelTrainer:
    def test_trainer_initialization(self):
        """Test inisialisasi HybridModelTrainer"""
        trainer = HybridModelTrainer()
        assert trainer is not None
        assert trainer.config is not None
        assert trainer.pipeline is not None
        assert trainer.model_serializer is not None
        
    def test_save_trained_models(self, mocker, tmp_path):
        """Test menyimpan trained models"""
        trainer = HybridModelTrainer()
        
        # Mock results
        trainer.results = {
            'individual_results': {},
            'hybrid_results': {
                'accuracy': 0.85, 'precision': 0.82, 'recall': 0.80, 
                'f1': 0.81, 'roc_auc': 0.88, 'threshold': 0.45
            },
            'models': {
                'gb_model': mocker.Mock(),
                'nn_model': mocker.Mock(),
                'embedding_model': mocker.Mock(),
                'nb_model': mocker.Mock()
            },
            'feature_importance': pd.DataFrame({
                'feature': ['feat1', 'feat2'],
                'importance': [0.6, 0.4]
            })
        }
        
        # Mock serializer
        mock_serializer = mocker.Mock()
        mock_serializer.save_all_models.return_value = {
            'gradient_boosting': 'path/to/gb.pkl',
            'neural_network': 'path/to/nn.h5'
        }
        trainer.model_serializer = mock_serializer
        
        scaler = mocker.Mock()
        selector = mocker.Mock()
        final_features = ['feat1', 'feat2']
        
        saved_paths = trainer.save_trained_models(scaler, selector, final_features)
        
        assert saved_paths is not None
        mock_serializer.save_all_models.assert_called_once()