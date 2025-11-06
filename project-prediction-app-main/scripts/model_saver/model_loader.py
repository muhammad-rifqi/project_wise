# scripts/model_saver/model_loader.py
import pickle
import joblib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

class ModelLoader:
    def __init__(self, base_dir="model"):
        self.base_dir = base_dir
        self.trained_models_dir = os.path.join(base_dir, "trained_models")
        self.preprocessors_dir = os.path.join(base_dir, "preprocessors")
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] ModelLoader: {message}")
        
    def load_gradient_boosting(self, model_name="gradient_boosting"):
        """Memuat model Gradient Boosting"""
        self._log(f"Memuat model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} tidak ditemukan: {model_path}")
            
        model = joblib.load(model_path)
        self._log(f"Model {model_name} berhasil dimuat")
        return model
        
    def load_neural_network(self, model_name="neural_network"):
        """Memuat model Neural Network"""
        self._log(f"Memuat model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} tidak ditemukan: {model_path}")
            
        model = load_model(model_path)
        self._log(f"Model {model_name} berhasil dimuat")
        return model
        
    def load_embedding_model(self, model_name="embedding_model"):
        """Memuat embedding model"""
        self._log(f"Memuat model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} tidak ditemukan: {model_path}")
            
        model = load_model(model_path)
        self._log(f"Model {model_name} berhasil dimuat")
        return model
        
    def load_naive_bayes(self, model_name="naive_bayes"):
        """Memuat model Naive Bayes"""
        self._log(f"Memuat model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} tidak ditemukan: {model_path}")
            
        model = joblib.load(model_path)
        self._log(f"Model {model_name} berhasil dimuat")
        return model
        
    def load_scaler(self, scaler_name="standard_scaler"):
        """Memuat scaler"""
        self._log(f"Memuat scaler {scaler_name}")
        
        scaler_path = os.path.join(self.preprocessors_dir, f"{scaler_name}.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler {scaler_name} tidak ditemukan: {scaler_path}")
            
        scaler = joblib.load(scaler_path)
        self._log(f"Scaler {scaler_name} berhasil dimuat")
        return scaler
        
    def load_feature_selector(self, selector_name="feature_selector"):
        """Memuat feature selector"""
        self._log(f"Memuat feature selector {selector_name}")
        
        selector_path = os.path.join(self.preprocessors_dir, f"{selector_name}.pkl")
        if not os.path.exists(selector_path):
            raise FileNotFoundError(f"Feature selector {selector_name} tidak ditemukan: {selector_path}")
            
        selector = joblib.load(selector_path)
        self._log(f"Feature selector {selector_name} berhasil dimuat")
        return selector
        
    def load_feature_names(self, filename="feature_names.json"):
        """Memuat nama features"""
        self._log("Memuat feature names")
        
        feature_path = os.path.join(self.preprocessors_dir, filename)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature names tidak ditemukan: {feature_path}")
            
        with open(feature_path, 'r') as f:
            feature_names = json.load(f)
        self._log("Feature names berhasil dimuat")
        return feature_names
        
    def load_model_config(self, filename="model_config.json"):
        """Memuat konfigurasi model"""
        self._log("Memuat model configuration")
        
        config_path = os.path.join(self.base_dir, filename)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config tidak ditemukan: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        self._log("Model config berhasil dimuat")
        return config
        
    def load_training_results(self, filename="training_results.json"):
        """Memuat hasil training"""
        self._log("Memuat training results")
        
        results_path = os.path.join(self.base_dir, filename)
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Training results tidak ditemukan: {results_path}")
            
        with open(results_path, 'r') as f:
            results = json.load(f)
        self._log("Training results berhasil dimuat")
        return results
        
    def load_all_models(self):
        """Memuat semua model dan komponen terkait"""
        self._log("=== MEMUAT SEMUA MODEL DAN KOMPONEN ===")
        
        loaded_models = {}
        
        try:
            # Load models
            loaded_models['gradient_boosting'] = self.load_gradient_boosting()
            loaded_models['neural_network'] = self.load_neural_network()
            loaded_models['embedding_model'] = self.load_embedding_model()
            loaded_models['naive_bayes'] = self.load_naive_bayes()
            
            # Load preprocessors
            loaded_models['scaler'] = self.load_scaler()
            loaded_models['feature_selector'] = self.load_feature_selector()
            loaded_models['feature_names'] = self.load_feature_names()
            
            # Load config and results
            loaded_models['config'] = self.load_model_config()
            loaded_models['training_results'] = self.load_training_results()
            
            self._log("=== SEMUA MODEL BERHASIL DIMUAT ===")
            return loaded_models
            
        except FileNotFoundError as e:
            self._log(f"ERROR: {e}")
            raise