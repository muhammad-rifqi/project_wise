# tests/test_feature_engineer.py
import pytest
import pandas as pd
import numpy as np
from scripts.feature_engineer.transformer import FeatureTransformer
from scripts.feature_engineer.aggregator import DataAggregator
from scripts.feature_engineer.visualizer import DataVisualizer

class TestFeatureTransformer:
    def test_transformer_initialization(self):
        """Test inisialisasi FeatureTransformer"""
        transformer = FeatureTransformer()
        assert transformer is not None
        
    def test_encode_categorical_features(self, sample_model_data):
        """Test encoding categorical features"""
        transformer = FeatureTransformer()
        encoded_data, categorical_features = transformer.encode_categorical_features(sample_model_data)
        
        assert encoded_data is not None
        assert 'scale_encoded' in encoded_data.columns
        assert any('type_' in col for col in encoded_data.columns)
        assert isinstance(categorical_features, list)
        
    def test_create_target_variable(self):
        """Test pembuatan target variable"""
        transformer = FeatureTransformer()
        
        sample_data = pd.DataFrame({
            'status_project': ['success', 'failed', 'success', 'failed']
        })
        
        result = transformer.create_target_variable(sample_data)
        
        assert 'project_success' in result.columns
        assert result['project_success'].dtype == int
        assert result['project_success'].sum() == 2  # 2 success
        
    def test_select_final_features(self, sample_model_data):
        """Test pemilihan final features"""
        transformer = FeatureTransformer()
        
        # Encode data terlebih dahulu
        encoded_data, categorical_features = transformer.encode_categorical_features(sample_model_data)
        
        numerical_features = [
            'duration_months', 'total_development_cost', 'additional_costs',
            'total_team_size', 'avg_expertise', 'max_expertise', 'avg_salary',
            'total_risks', 'high_impact_risks', 'high_likelihood_risks',
            'unique_tech_types', 'total_tools', 'sdlc_method_id', 'scale_encoded'
        ]
        
        X, y, final_features = transformer.select_final_features(
            encoded_data, numerical_features, categorical_features
        )
        
        assert X is not None
        assert y is not None
        assert len(final_features) > 0
        assert X.shape[1] == len(final_features)

class TestDataAggregator:
    def test_aggregator_initialization(self):
        """Test inisialisasi DataAggregator"""
        aggregator = DataAggregator()
        assert aggregator is not None
        
    def test_aggregate_team_data(self, sample_data_dict):
        """Test aggregasi data team"""
        aggregator = DataAggregator()
        team_agg = aggregator.aggregate_team_data(
            sample_data_dict['team_members'], 
            sample_data_dict['allocations']
        )
        
        assert team_agg is not None
        assert 'project_id' in team_agg.columns
        assert 'total_team_size' in team_agg.columns
        assert 'avg_expertise' in team_agg.columns
        
    def test_aggregate_risk_data(self, sample_data_dict):
        """Test aggregasi data risks"""
        aggregator = DataAggregator()
        risk_agg = aggregator.aggregate_risk_data(sample_data_dict['risks'])
        
        assert risk_agg is not None
        assert 'project_id' in risk_agg.columns
        assert 'total_risks' in risk_agg.columns
        assert 'high_impact_risks' in risk_agg.columns
        assert 'high_likelihood_risks' in risk_agg.columns
        
    def test_aggregate_technology_data(self, sample_data_dict):
        """Test aggregasi data technologies"""
        aggregator = DataAggregator()
        tech_agg = aggregator.aggregate_technology_data(sample_data_dict['project_technologies'])
        
        assert tech_agg is not None
        assert 'project_id' in tech_agg.columns
        assert 'unique_tech_types' in tech_agg.columns
        assert 'total_tools' in tech_agg.columns
        
    def test_merge_all_datasets(self, sample_data_dict):
        """Test penggabungan dataset"""
        aggregator = DataAggregator()
        
        # Buat aggregated data terlebih dahulu
        team_agg = aggregator.aggregate_team_data(
            sample_data_dict['team_members'],
            sample_data_dict['allocations']
        )
        risk_agg = aggregator.aggregate_risk_data(sample_data_dict['risks'])
        tech_agg = aggregator.aggregate_technology_data(sample_data_dict['project_technologies'])
        
        merged_data = aggregator.merge_all_datasets(
            sample_data_dict['projects'], 
            sample_data_dict['allocations'], 
            team_agg, risk_agg, tech_agg
        )
        
        assert merged_data is not None
        assert len(merged_data) > 0
        
    def test_handle_missing_values(self):
        """Test handling missing values"""
        aggregator = DataAggregator()
        
        # Buat sample data dengan kolom yang sesuai dengan implementasi
        sample_data = pd.DataFrame({
            'total_risks': [2, None, 1],
            'high_impact_risks': [1, None, 0],  # ✅ TAMBAHKAN KOLOM INI
            'high_likelihood_risks': [1, None, 1],  # ✅ TAMBAHKAN KOLOM INI
            'unique_tech_types': [2, 1, None],  # ✅ TAMBAHKAN KOLOM INI
            'total_tools': [2, 1, None],  # ✅ TAMBAHKAN KOLOM INI
            'total_team_size': [5, 3, None],
            'avg_expertise': [5.5, None, 7.0],  # ✅ TAMBAHKAN KOLOM INI
            'max_expertise': [6, 4, None],  # ✅ TAMBAHKAN KOLOM INI
            'avg_salary': [5000, 6000, None]
        })
        
        result = aggregator.handle_missing_values(sample_data)
        
        assert result is not None
        # Test bahwa semua missing values sudah dihandle
        for col in result.columns:
            assert result[col].isnull().sum() == 0

class TestDataVisualizer:
    def test_visualizer_initialization(self):
        """Test inisialisasi DataVisualizer"""
        visualizer = DataVisualizer(output_dir="test_charts")
        assert visualizer is not None
        assert visualizer.output_dir == "test_charts"
        
    def test_plot_target_distribution(self, sample_model_data, tmp_path):
        """Test plotting target distribution"""
        visualizer = DataVisualizer(output_dir=str(tmp_path))
        
        # Test bahwa method bisa dipanggil tanpa error
        try:
            visualizer.plot_target_distribution(sample_model_data)
            assert True
        except Exception as e:
            pytest.fail(f"plot_target_distribution failed with {e}")