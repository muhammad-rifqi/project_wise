# scripts/train_hybrid_model/trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# Absolute imports
from scripts.data_preprocessor.loader import DataLoader
from scripts.data_preprocessor.validator import DataValidator
from scripts.feature_engineer.transformer import FeatureTransformer
from scripts.feature_engineer.aggregator import DataAggregator
from scripts.feature_engineer.visualizer import DataVisualizer
from scripts.model_builder.gradient_boosting import GradientBoostingTrainer
from scripts.model_builder.neural_network import NeuralNetworkTrainer
from scripts.model_builder.naive_bayes import NaiveBayesTrainer
from scripts.model_builder.hybrid_fusion import HybridFeatureFusion
from scripts.model_evaluator.metrics import ModelMetrics
from scripts.model_evaluator.visualizer import EvaluationVisualizer
from scripts.model_evaluator.comparator import ModelComparator
from scripts.model_saver import ModelSerializer

from .config import ModelConfig

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
        
        # Visualize data - sekarang akan disimpan ke folder
        visualizer = DataVisualizer()
        visualizer.plot_target_distribution(model_data)
        visualizer.plot_feature_distributions(model_data, numerical_features)
        visualizer.plot_correlation_matrix(model_data, numerical_features)
        
        return X, y, final_features, model_data

    def run_model_training(self, X_train_scaled, X_test_scaled, y_train, y_test, final_features):
        """Menjalankan fase model training dan evaluasi"""
        self._log("=== FASE 4: MODEL TRAINING ===")
        
        # 1. Train Gradient Boosting
        gb_trainer = GradientBoostingTrainer()
        gb_model, selector, X_train_gb = gb_trainer.train_model(X_train_scaled, y_train)  # Kembalikan selector juga
        X_test_gb = selector.transform(X_test_scaled)
        feature_importance = gb_trainer.get_feature_importance(final_features)
        
        # 2. Train Neural Network
        nn_trainer = NeuralNetworkTrainer()
        nn_model, embedding_model = nn_trainer.build_model(X_train_scaled.shape[1])
        nn_history = nn_trainer.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        X_train_ann = nn_trainer.extract_embeddings(X_train_scaled)
        X_test_ann = nn_trainer.extract_embeddings(X_test_scaled)
        
        # 3. Create hybrid features and train Naive Bayes
        fusion = HybridFeatureFusion()
        X_train_hybrid = fusion.create_hybrid_features(X_train_gb, X_train_ann)
        X_test_hybrid = fusion.create_hybrid_features(X_test_gb, X_test_ann)
        
        nb_trainer = NaiveBayesTrainer()
        nb_model = nb_trainer.train_model(X_train_hybrid, y_train)
        
        # 4. Evaluate individual models
        metrics_calculator = ModelMetrics()
        visualizer = EvaluationVisualizer()  # Ini akan menyimpan chart ke folder
        comparator = ModelComparator()
        
        # Evaluate GB
        y_pred_gb = gb_model.predict(X_test_scaled)
        y_pred_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]
        gb_metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred_gb, y_pred_proba_gb)
        
        # Evaluate ANN
        y_pred_ann = (nn_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
        y_pred_proba_ann = nn_model.predict(X_test_scaled).flatten()
        ann_metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred_ann, y_pred_proba_ann)
        
        # Evaluate NB
        y_pred_nb, y_pred_proba_nb = nb_trainer.predict(X_test_hybrid)
        nb_metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred_nb, y_pred_proba_nb[:, 1])
        
        individual_results = {
            'gradient_boosting': gb_metrics,
            'neural_network': ann_metrics,
            'naive_bayes': nb_metrics
        }
        
        # 5. Create hybrid weighted model
        weighted_proba = fusion.create_weighted_predictions(
            y_pred_proba_gb, y_pred_proba_ann, y_pred_proba_nb[:, 1],
            weights=self.config.MODEL_WEIGHTS
        )
        
        optimal_threshold, best_f1 = metrics_calculator.find_optimal_threshold(weighted_proba, y_test)
        y_pred_hybrid = (weighted_proba >= optimal_threshold).astype(int)
        
        hybrid_metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred_hybrid, weighted_proba)
        hybrid_metrics['threshold'] = optimal_threshold
        
        # 6. Visualize results - semua akan disimpan ke folder
        visualizer.plot_roc_curves(individual_results)
        visualizer.plot_training_history(nn_history, "Neural Network")
        visualizer.plot_metrics_comparison({**individual_results, 'hybrid': hybrid_metrics})
        
        # Plot confusion matrices
        for model_name, results in individual_results.items():
            visualizer.plot_confusion_matrix(results['confusion_matrix'], model_name)
        visualizer.plot_confusion_matrix(hybrid_metrics['confusion_matrix'], 'Hybrid Model')
        
        # Plot feature importance
        visualizer.plot_feature_importance(feature_importance)
        
        # 7. Comparative analysis
        comparator.print_detailed_comparison(individual_results, hybrid_metrics)
        
        return {
            'individual_results': individual_results,
            'hybrid_results': hybrid_metrics,
            'models': {
                'gb_model': gb_model,
                'nn_model': nn_model,
                'nb_model': nb_model,
                'embedding_model': embedding_model,
                'selector': selector  # SIMPAN SELECTOR
            },
            'feature_importance': feature_importance
        }

class HybridModelTrainer:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.pipeline = TrainingPipeline(self.config)
        self.results = None
        self.start_time = None
        self.model_serializer = ModelSerializer()
        
    def _log(self, message):
        """Log step dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] {message}")
        
    def save_trained_models(self, scaler, selector, final_features):
        """Menyimpan semua model yang sudah di-training"""
        self._log("=== MENYIMPAN MODEL YANG SUDAH DI-TRAINING ===")
        
        if self.results is None:
            self._log("ERROR: Tidak ada model yang bisa disimpan, jalankan training terlebih dahulu")
            return None
            
        try:
            saved_paths = self.model_serializer.save_all_models(
                results=self.results,
                config=self.config,
                feature_names=final_features,
                scaler=scaler,
                selector=selector
            )
            
            self._log("Semua model berhasil disimpan di folder 'model/'")
            return saved_paths
            
        except Exception as e:
            self._log(f"ERROR dalam menyimpan model: {e}")
            return None
        
    def run_pipeline(self):
        """Menjalankan seluruh pipeline training model hybrid"""
        self.start_time = time.time()
        self._log("HYBRID MODEL TRAINING PIPELINE DIMULAI")
        
        try:
            # 1. Data Processing
            data_dict = self.pipeline.run_data_processing()
            
            # 2. Feature Engineering
            X, y, final_features, model_data = self.pipeline.run_feature_engineering(data_dict)
            
            # 3. Data Preparation
            self._log("=== FASE 3: DATA PREPARATION ===")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, 
                random_state=self.config.RANDOM_STATE, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self._log(f"Data siap: Training set {X_train_scaled.shape}, Test set {X_test_scaled.shape}")
            
            # 4. Model Training & Evaluation
            self.results = self.pipeline.run_model_training(
                X_train_scaled, X_test_scaled, y_train, y_test, final_features
            )
            
            # 5. Simpan Model
            self._log("=== FASE 5: MENYIMPAN MODEL ===")
            # Dapatkan selector dari results
            selector = self.results['models']['selector']
            saved_paths = self.save_trained_models(scaler, selector, final_features)
            
            # Calculate total execution time
            total_time = time.time() - self.start_time
            self._log(f"Pipeline selesai - Total waktu eksekusi: {total_time:.2f} detik")
            
            return {
                'results': self.results,
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            self._log(f"ERROR dalam pipeline: {e}")
            raise

def main():
    """Main function untuk menjalankan training pipeline"""
    print("=" * 70)
    print("HYBRID MACHINE LEARNING MODEL TRAINING")
    print("=" * 70)
    
    trainer = HybridModelTrainer()
    results = trainer.run_pipeline()
    
    if results:
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE BERHASIL DISELESAIKAN")
        print("=" * 70)
        
        # Tampilkan hasil terbaik
        hybrid_result = results['results']['hybrid_results']
        print("\nHASIL TERBAIK (Hybrid Weighted Voting Model):")
        print(f"Accuracy:  {hybrid_result['accuracy']:.4f}")
        print(f"Precision: {hybrid_result['precision']:.4f}")
        print(f"Recall:    {hybrid_result['recall']:.4f}")
        print(f"F1-Score:  {hybrid_result['f1']:.4f}")
        print(f"ROC-AUC:   {hybrid_result['roc_auc']:.4f}")
        print(f"Optimal Threshold: {hybrid_result['threshold']:.2f}")
        
        # Tampilkan info model yang disimpan
        if results.get('saved_paths'):
            print(f"\nModel disimpan di folder: model/")
            for model_name, path in results['saved_paths'].items():
                print(f"{model_name}: {path}")

if __name__ == "__main__":
    main()