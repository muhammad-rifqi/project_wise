# scripts/model_evaluator/metrics.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import numpy as np
from datetime import datetime

class ModelMetrics:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] ModelMetrics: {message}")
        
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Menghitung semua metrics evaluasi"""
        self._log("Menghitung metrics evaluasi")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr
        }
        
        self._log(f"Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
        
        return metrics
        
    def find_optimal_threshold(self, y_pred_proba, y_true):
        """Mencari threshold optimal berdasarkan F1-score"""
        self._log("Mencari threshold optimal")
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.3, 0.71, 0.01):
            y_pred_temp = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_temp)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        self._log(f"Threshold optimal: {best_threshold:.2f} (F1={best_f1:.4f})")
        return best_threshold, best_f1