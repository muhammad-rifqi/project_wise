# tests/test_model_builder.py
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from scripts.model_builder.gradient_boosting import GradientBoostingTrainer
from scripts.model_builder.neural_network import NeuralNetworkTrainer
from scripts.model_builder.naive_bayes import NaiveBayesTrainer
from scripts.model_builder.hybrid_fusion import HybridFeatureFusion

class TestGradientBoostingTrainer:
    def test_gb_initialization(self):
        """Test inisialisasi GradientBoostingTrainer"""
        trainer = GradientBoostingTrainer()
        assert trainer is not None
        assert trainer.model is None
        assert trainer.selector is None
        
    def test_train_model(self, sample_training_data):
        """Test training Gradient Boosting model"""
        X, y = sample_training_data
        trainer = GradientBoostingTrainer()
        
        model, selector, X_selected = trainer.train_model(X, y)
        
        assert model is not None
        assert selector is not None
        assert X_selected.shape[1] <= X.shape[1]  # Feature selection mengurangi dimensi
        
    def test_get_feature_importance(self, sample_training_data, sample_feature_importance):
        """Test mendapatkan feature importance"""
        X, y = sample_training_data
        trainer = GradientBoostingTrainer()
        model, selector, X_selected = trainer.train_model(X, y)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = trainer.get_feature_importance(feature_names)
        
        assert importance_df is not None
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(feature_names)

class TestNeuralNetworkTrainer:
    def test_nn_initialization(self):
        """Test inisialisasi NeuralNetworkTrainer"""
        trainer = NeuralNetworkTrainer()
        assert trainer is not None
        assert trainer.model is None
        assert trainer.embedding_model is None
        
    def test_build_model(self):
        """Test membangun model neural network"""
        trainer = NeuralNetworkTrainer()
        input_dim = 10
        
        model, embedding_model = trainer.build_model(input_dim)
        
        assert model is not None
        assert embedding_model is not None
        assert model.input_shape[1] == input_dim
        
    def test_train_model(self, sample_training_data):
        """Test training neural network"""
        X, y = sample_training_data
        trainer = NeuralNetworkTrainer()
        
        model, embedding_model = trainer.build_model(X.shape[1])
        history = trainer.train_model(X, y, X, y)  # Using same data for train/test for testing
        
        assert history is not None
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        
    def test_extract_embeddings(self, sample_training_data):
        """Test ekstraksi embeddings"""
        X, y = sample_training_data
        trainer = NeuralNetworkTrainer()
        
        model, embedding_model = trainer.build_model(X.shape[1])
        trainer.train_model(X, y, X, y)
        
        embeddings = trainer.extract_embeddings(X)
        
        assert embeddings is not None
        assert embeddings.shape[0] == X.shape[0]
        assert embeddings.shape[1] == 16  # Sesuai dengan embedding layer size

class TestNaiveBayesTrainer:
    def test_nb_initialization(self):
        """Test inisialisasi NaiveBayesTrainer"""
        trainer = NaiveBayesTrainer()
        assert trainer is not None
        assert trainer.model is None
        
    def test_train_model(self, sample_training_data):
        """Test training Naive Bayes model"""
        X, y = sample_training_data
        trainer = NaiveBayesTrainer()
        
        model = trainer.train_model(X, y)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
    def test_predict(self, sample_training_data):
        """Test prediction dengan Naive Bayes"""
        X, y = sample_training_data
        trainer = NaiveBayesTrainer()
        
        model = trainer.train_model(X, y)
        y_pred, y_proba = trainer.predict(X)
        
        assert y_pred is not None
        assert y_proba is not None
        assert len(y_pred) == len(y)
        assert y_proba.shape[0] == len(y)

class TestHybridFeatureFusion:
    def test_fusion_initialization(self):
        """Test inisialisasi HybridFeatureFusion"""
        fusion = HybridFeatureFusion()
        assert fusion is not None
        
    def test_create_hybrid_features(self):
        """Test pembuatan hybrid features"""
        fusion = HybridFeatureFusion()
        
        gb_features = np.random.randn(100, 5)
        ann_embeddings = np.random.randn(100, 10)
        
        hybrid_features = fusion.create_hybrid_features(gb_features, ann_embeddings)
        
        assert hybrid_features is not None
        assert hybrid_features.shape[0] == 100
        assert hybrid_features.shape[1] == 15  # 5 + 10
        
    def test_create_weighted_predictions(self):
        """Test pembuatan weighted predictions"""
        fusion = HybridFeatureFusion()
        
        gb_proba = np.random.rand(100)
        ann_proba = np.random.rand(100)
        nb_proba = np.random.rand(100)
        
        weighted_proba = fusion.create_weighted_predictions(gb_proba, ann_proba, nb_proba)
        
        assert weighted_proba is not None
        assert len(weighted_proba) == 100
        assert np.all(weighted_proba >= 0)
        assert np.all(weighted_proba <= 1)