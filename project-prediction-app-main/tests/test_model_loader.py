# tests/test_model_loader.py
import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import tempfile

# Tambahkan path ke sys.path untuk bisa import dari scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf
from scripts.model_saver.model_serializer import ModelSerializer
from scripts.model_saver.model_loader import ModelLoader

class TestModelLoader:
    def test_loader_initialization(self, tmp_path):
        """Test inisialisasi ModelLoader"""
        loader = ModelLoader(base_dir=str(tmp_path))
        assert loader is not None
        assert loader.base_dir == str(tmp_path)
        
    def test_load_gradient_boosting(self, tmp_path):
        """Test memuat model Gradient Boosting"""
        # Setup: Simpan model terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        serializer.save_gradient_boosting(model)
        
        # Test: Load model
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_model = loader.load_gradient_boosting()
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
    def test_load_neural_network(self, tmp_path):
        """Test memuat model Neural Network"""
        # Setup: Simpan model terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Untuk testing, kita hanya perlu menyimpan model tanpa training
        model_path = serializer.save_neural_network(model)
        
        # Test: Load model
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_model = loader.load_neural_network()
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
    def test_load_naive_bayes(self, tmp_path):
        """Test memuat model Naive Bayes"""
        # Setup: Simpan model terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        model = GaussianNB()
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        serializer.save_naive_bayes(model)
        
        # Test: Load model
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_model = loader.load_naive_bayes()
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
    def test_load_scaler(self, tmp_path):
        """Test memuat scaler"""
        # Setup: Simpan scaler terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        scaler = StandardScaler()
        X = np.random.randn(100, 5)
        scaler.fit(X)
        serializer.save_scaler(scaler)
        
        # Test: Load scaler
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_scaler = loader.load_scaler()
        
        assert loaded_scaler is not None
        assert hasattr(loaded_scaler, 'transform')
        
    def test_load_feature_selector(self, tmp_path):
        """Test memuat feature selector"""
        # Setup: Simpan feature selector terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        
        # Buat model dan selector dummy
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        selector = SelectFromModel(model, prefit=True)
        serializer.save_feature_selector(selector)
        
        # Test: Load feature selector
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_selector = loader.load_feature_selector()
        
        assert loaded_selector is not None
        assert hasattr(loaded_selector, 'transform')
        
    def test_load_feature_names(self, tmp_path):
        """Test memuat feature names"""
        # Setup: Simpan feature names terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        feature_names = [f'feature_{i}' for i in range(10)]
        serializer.save_feature_names(feature_names)
        
        # Test: Load feature names
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_features = loader.load_feature_names()
        
        assert loaded_features is not None
        assert loaded_features == feature_names
        
    def test_load_model_config(self, tmp_path):
        """Test memuat model config"""
        # Setup: Simpan model config terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        
        # Buat config dummy
        class MockConfig:
            RANDOM_STATE = 42
            TEST_SIZE = 0.2
            ANN_EPOCHS = 100
            ANN_BATCH_SIZE = 32
            MODEL_WEIGHTS = {'gradient_boosting': 0.5, 'neural_network': 0.3, 'naive_bayes': 0.2}
            FEATURE_SELECTION_THRESHOLD = 'median'
            NAIVE_BAYES_VAR_SMOOTHING = 1e-2
            
        config = MockConfig()
        serializer.save_model_config(config)
        
        # Test: Load model config
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_config = loader.load_model_config()
        
        assert loaded_config is not None
        assert loaded_config['RANDOM_STATE'] == 42
        assert loaded_config['TEST_SIZE'] == 0.2
        
    def test_load_training_results(self, tmp_path):
        """Test memuat training results"""
        # Setup: Simpan training results terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        
        # Buat results dummy dengan DataFrame untuk feature_importance
        results = {
            'individual_results': {
                'gradient_boosting': {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.80,
                    'f1': 0.81,
                    'roc_auc': 0.88
                },
                'neural_network': {
                    'accuracy': 0.83,
                    'precision': 0.80,
                    'recall': 0.78,
                    'f1': 0.79,
                    'roc_auc': 0.86
                },
                'naive_bayes': {
                    'accuracy': 0.80,
                    'precision': 0.78,
                    'recall': 0.75,
                    'f1': 0.76,
                    'roc_auc': 0.84
                }
            },
            'hybrid_results': {
                'accuracy': 0.87,
                'precision': 0.84,
                'recall': 0.82,
                'f1': 0.83,
                'roc_auc': 0.90,
                'threshold': 0.45
            },
            'feature_importance': pd.DataFrame({
                'feature': ['feat1', 'feat2', 'feat3'],
                'importance': [0.5, 0.3, 0.2]
            })
        }
        
        serializer.save_training_results(results)
        
        # Test: Load training results
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_results = loader.load_training_results()
        
        assert loaded_results is not None
        assert 'individual_results' in loaded_results
        assert 'hybrid_results' in loaded_results
        assert loaded_results['hybrid_results']['accuracy'] == 0.87
        
    def test_load_all_models(self, tmp_path):
        """Test memuat semua model sekaligus"""
        # Setup: Simpan semua model terlebih dahulu
        serializer = ModelSerializer(base_dir=str(tmp_path))
        
        # Buat dan simpan komponen dummy
        gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        gb_model.fit(X, y)
        
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        nn_model.compile(optimizer='adam', loss='binary_crossentropy')
        
        embedding_model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(5,))
        ])
        
        nb_model = GaussianNB()
        nb_model.fit(X, y)
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        selector = SelectFromModel(gb_model, prefit=True)
        
        feature_names = [f'feature_{i}' for i in range(5)]
        
        class MockConfig:
            RANDOM_STATE = 42
            TEST_SIZE = 0.2
            ANN_EPOCHS = 100
            ANN_BATCH_SIZE = 32
            MODEL_WEIGHTS = {'gradient_boosting': 0.5, 'neural_network': 0.3, 'naive_bayes': 0.2}
            FEATURE_SELECTION_THRESHOLD = 'median'
            NAIVE_BAYES_VAR_SMOOTHING = 1e-2
            
        config = MockConfig()
        
        # Buat results untuk save_all_models
        results = {
            'individual_results': {
                'gradient_boosting': {
                    'accuracy': 0.85, 'precision': 0.82, 'recall': 0.80, 
                    'f1': 0.81, 'roc_auc': 0.88
                },
                'neural_network': {
                    'accuracy': 0.83, 'precision': 0.80, 'recall': 0.78, 
                    'f1': 0.79, 'roc_auc': 0.86
                },
                'naive_bayes': {
                    'accuracy': 0.80, 'precision': 0.78, 'recall': 0.75, 
                    'f1': 0.76, 'roc_auc': 0.84
                }
            },
            'hybrid_results': {
                'accuracy': 0.87, 'precision': 0.84, 'recall': 0.82, 
                'f1': 0.83, 'roc_auc': 0.90, 'threshold': 0.45
            },
            'models': {
                'gb_model': gb_model,
                'nn_model': nn_model,
                'embedding_model': embedding_model,
                'nb_model': nb_model
            },
            'feature_importance': pd.DataFrame({
                'feature': feature_names,
                'importance': np.random.rand(5)
            })
        }
        
        serializer.save_all_models(results, config, feature_names, scaler, selector)
        
        # Test: Load semua model
        loader = ModelLoader(base_dir=str(tmp_path))
        loaded_models = loader.load_all_models()
        
        assert loaded_models is not None
        assert isinstance(loaded_models, dict)
        assert 'gradient_boosting' in loaded_models
        assert 'neural_network' in loaded_models
        assert 'scaler' in loaded_models
        assert 'feature_names' in loaded_models
        assert 'config' in loaded_models
        assert 'training_results' in loaded_models

    def test_file_not_found_error(self, tmp_path):
        """Test error handling ketika file tidak ditemukan"""
        loader = ModelLoader(base_dir=str(tmp_path))
        
        # Test bahwa FileNotFoundError di-raise ketika file tidak ada
        with pytest.raises(FileNotFoundError):
            loader.load_gradient_boosting()
            
        with pytest.raises(FileNotFoundError):
            loader.load_scaler()

# Menjalankan tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])