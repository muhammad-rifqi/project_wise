# scripts/model_saver/model_serializer.py
import pickle
import joblib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf

class ModelSerializer:
    def __init__(self, base_dir="model"):
        self.base_dir = base_dir
        self.trained_models_dir = os.path.join(base_dir, "trained_models")
        self.preprocessors_dir = os.path.join(base_dir, "preprocessors")
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Membuat direktori jika belum ada"""
        os.makedirs(self.trained_models_dir, exist_ok=True)
        os.makedirs(self.preprocessors_dir, exist_ok=True)
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] ModelSerializer: {message}")
        
    def save_gradient_boosting(self, model, model_name="gradient_boosting"):
        """Menyimpan model Gradient Boosting"""
        self._log(f"Menyimpan model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        self._log(f"Model {model_name} disimpan: {model_path}")
        
        return model_path
        
    def save_neural_network(self, model, model_name="neural_network"):
        """Menyimpan model Neural Network"""
        self._log(f"Menyimpan model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.h5")
        model.save(model_path)
        self._log(f"Model {model_name} disimpan: {model_path}")
        
        return model_path
        
    def save_embedding_model(self, model, model_name="embedding_model"):
        """Menyimpan embedding model"""
        self._log(f"Menyimpan model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.h5")
        model.save(model_path)
        self._log(f"Model {model_name} disimpan: {model_path}")
        
        return model_path
        
    def save_naive_bayes(self, model, model_name="naive_bayes"):
        """Menyimpan model Naive Bayes"""
        self._log(f"Menyimpan model {model_name}")
        
        model_path = os.path.join(self.trained_models_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        self._log(f"Model {model_name} disimpan: {model_path}")
        
        return model_path
        
    def save_scaler(self, scaler, scaler_name="standard_scaler"):
        """Menyimpan scaler"""
        self._log(f"Menyimpan scaler {scaler_name}")
        
        scaler_path = os.path.join(self.preprocessors_dir, f"{scaler_name}.pkl")
        joblib.dump(scaler, scaler_path)
        self._log(f"Scaler {scaler_name} disimpan: {scaler_path}")
        
        return scaler_path
        
    def save_feature_selector(self, selector, selector_name="feature_selector"):
        """Menyimpan feature selector"""
        self._log(f"Menyimpan feature selector {selector_name}")
        
        selector_path = os.path.join(self.preprocessors_dir, f"{selector_name}.pkl")
        joblib.dump(selector, selector_path)
        self._log(f"Feature selector {selector_name} disimpan: {selector_path}")
        
        return selector_path
        
    def save_feature_names(self, feature_names, filename="feature_names.json"):
        """Menyimpan nama features"""
        self._log("Menyimpan feature names")
        
        feature_path = os.path.join(self.preprocessors_dir, filename)
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        self._log(f"Feature names disimpan: {feature_path}")
        
        return feature_path
        
    def save_model_config(self, config, filename="model_config.json"):
        """Menyimpan konfigurasi model"""
        self._log("Menyimpan model configuration")
        
        config_path = os.path.join(self.base_dir, filename)
        
        # Convert config to serializable format
        config_dict = {
            'RANDOM_STATE': config.RANDOM_STATE,
            'TEST_SIZE': config.TEST_SIZE,
            'ANN_EPOCHS': config.ANN_EPOCHS,
            'ANN_BATCH_SIZE': config.ANN_BATCH_SIZE,
            'MODEL_WEIGHTS': config.MODEL_WEIGHTS,
            'FEATURE_SELECTION_THRESHOLD': config.FEATURE_SELECTION_THRESHOLD,
            'NAIVE_BAYES_VAR_SMOOTHING': config.NAIVE_BAYES_VAR_SMOOTHING
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self._log(f"Model config disimpan: {config_path}")
        
        return config_path
        
    def save_training_results(self, results, filename="training_results.json"):
        """Menyimpan hasil training"""
        self._log("Menyimpan training results")
        
        results_path = os.path.join(self.base_dir, filename)
        
        # Convert results to serializable format
        results_dict = {
            'individual_results': {},
            'hybrid_results': {
                'accuracy': float(results['hybrid_results']['accuracy']),
                'precision': float(results['hybrid_results']['precision']),
                'recall': float(results['hybrid_results']['recall']),
                'f1': float(results['hybrid_results']['f1']),
                'roc_auc': float(results['hybrid_results']['roc_auc']),
                'threshold': float(results['hybrid_results']['threshold'])
            },
            'feature_importance': results['feature_importance'].to_dict('records')
        }
        
        # Convert individual results
        for model_name, metrics in results['individual_results'].items():
            results_dict['individual_results'][model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'roc_auc': float(metrics['roc_auc'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        self._log(f"Training results disimpan: {results_path}")
        
        return results_path
        
    def save_all_models(self, results, config, feature_names, scaler, selector):
        """Menyimpan semua model dan komponen terkait"""
        self._log("=== MENYIMPAN SEMUA MODEL DAN KOMPONEN ===")
        
        saved_paths = {}
        
        # Save models
        saved_paths['gradient_boosting'] = self.save_gradient_boosting(results['models']['gb_model'])
        saved_paths['neural_network'] = self.save_neural_network(results['models']['nn_model'])
        saved_paths['embedding_model'] = self.save_embedding_model(results['models']['embedding_model'])
        saved_paths['naive_bayes'] = self.save_naive_bayes(results['models']['nb_model'])
        
        # Save preprocessors
        saved_paths['scaler'] = self.save_scaler(scaler)
        saved_paths['feature_selector'] = self.save_feature_selector(selector)
        saved_paths['feature_names'] = self.save_feature_names(feature_names)
        
        # Save config and results
        saved_paths['config'] = self.save_model_config(config)
        saved_paths['training_results'] = self.save_training_results(results)
        
        self._log("=== SEMUA MODEL BERHASIL DISIMPAN ===")
        
        return saved_paths