# tests/test_model_evaluator.py
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scripts.model_evaluator.metrics import ModelMetrics
from scripts.model_evaluator.visualizer import EvaluationVisualizer
from scripts.model_evaluator.comparator import ModelComparator

class TestModelMetrics:
    def test_metrics_initialization(self):
        """Test inisialisasi ModelMetrics"""
        metrics = ModelMetrics()
        assert metrics is not None
        
    def test_calculate_all_metrics(self):
        """Test perhitungan semua metrics"""
        metrics_calc = ModelMetrics()
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.6, 0.8])
        
        results = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
        
        assert results is not None
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'roc_auc' in results
        assert 'confusion_matrix' in results
        
        # Test bahwa metrics dalam range yang benar
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1'] <= 1
        assert 0 <= results['roc_auc'] <= 1
        
    def test_find_optimal_threshold(self):
        """Test pencarian threshold optimal"""
        metrics_calc = ModelMetrics()
        
        y_pred_proba = np.random.rand(100)
        y_true = (y_pred_proba > 0.5).astype(int)
        
        optimal_threshold, best_f1 = metrics_calc.find_optimal_threshold(y_pred_proba, y_true)
        
        assert optimal_threshold is not None
        assert best_f1 is not None
        assert 0.3 <= optimal_threshold <= 0.7
        assert 0 <= best_f1 <= 1

class TestEvaluationVisualizer:
    def test_visualizer_initialization(self, tmp_path):
        """Test inisialisasi EvaluationVisualizer"""
        visualizer = EvaluationVisualizer(output_dir=str(tmp_path))
        assert visualizer is not None
        assert visualizer.output_dir == str(tmp_path)
        
    def test_plot_roc_curves(self, tmp_path):
        """Test plotting ROC curves"""
        visualizer = EvaluationVisualizer(output_dir=str(tmp_path))
        
        results_dict = {
            'model1': {
                'fpr': np.array([0.0, 0.5, 1.0]),
                'tpr': np.array([0.0, 0.7, 1.0]),
                'roc_auc': 0.85
            },
            'model2': {
                'fpr': np.array([0.0, 0.3, 1.0]),
                'tpr': np.array([0.0, 0.8, 1.0]),
                'roc_auc': 0.90
            }
        }
        
        try:
            visualizer.plot_roc_curves(results_dict)
            assert True
        except Exception as e:
            pytest.fail(f"plot_roc_curves failed with {e}")
            
    def test_plot_feature_importance(self, tmp_path, sample_feature_importance):
        """Test plotting feature importance"""
        visualizer = EvaluationVisualizer(output_dir=str(tmp_path))
        
        try:
            visualizer.plot_feature_importance(sample_feature_importance)
            assert True
        except Exception as e:
            pytest.fail(f"plot_feature_importance failed with {e}")

class TestModelComparator:
    def test_comparator_initialization(self):
        """Test inisialisasi ModelComparator"""
        comparator = ModelComparator()
        assert comparator is not None
        
    def test_create_comparison_table(self):
        """Test pembuatan comparison table"""
        comparator = ModelComparator()
        
        individual_results = {
            'model1': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.80,
                'f1': 0.81,
                'roc_auc': 0.88
            },
            'model2': {
                'accuracy': 0.82,
                'precision': 0.85,
                'recall': 0.75,
                'f1': 0.79,
                'roc_auc': 0.86
            }
        }
        
        hybrid_results = {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.82,
            'f1': 0.83,
            'roc_auc': 0.90
        }
        
        comparison_df = comparator.create_comparison_table(individual_results, hybrid_results)
        
        assert comparison_df is not None
        assert len(comparison_df) == 3  # 2 individual + 1 hybrid
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns
        
    def test_print_detailed_comparison(self, capsys):
        """Test detailed comparison printing"""
        comparator = ModelComparator()
        
        individual_results = {
            'gradient_boosting': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.80,
                'f1': 0.81,
                'roc_auc': 0.88
            }
        }
        
        hybrid_results = {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.82,
            'f1': 0.83,
            'roc_auc': 0.90
        }
        
        comparator.print_detailed_comparison(individual_results, hybrid_results)
        
        # Capture printed output
        captured = capsys.readouterr()
        assert "COMPARATIVE MODEL PERFORMANCE ANALYSIS" in captured.out
        assert "BEST PERFORMING MODELS" in captured.out