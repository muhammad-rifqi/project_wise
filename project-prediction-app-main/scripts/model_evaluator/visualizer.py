# scripts/model_evaluator/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

class EvaluationVisualizer:
    def __init__(self, output_dir="data/chart"):
        self.output_dir = output_dir
        self.chart_counter = 1
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Membuat folder output jika belum ada"""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] EvalVisualizer: {message}")
        
    def _save_chart(self, filename):
        """Menyimpan chart dengan penomoran otomatis"""
        chart_number = str(self.chart_counter).zfill(2)
        full_filename = f"{chart_number}_{filename}"
        full_path = os.path.join(self.output_dir, full_filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        self._log(f"Chart disimpan: {full_filename}")
        self.chart_counter += 1
        plt.close()  # Tutup plot untuk menghemat memory
        
    def plot_roc_curves(self, results_dict):
        """Plot ROC curves untuk multiple models"""
        self._log("Generating ROC curves comparison")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results_dict.items():
            plt.plot(result['fpr'], result['tpr'], 
                    label=f'{model_name} (AUC = {result["roc_auc"]:.4f})',
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_chart("roc_curves_comparison.png")
        
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix untuk model tertentu"""
        self._log(f"Generating confusion matrix untuk {model_name}")
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Failed', 'Predicted Success'],
                   yticklabels=['Actual Failed', 'Actual Success'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        self._save_chart(filename)
        
    def plot_metrics_comparison(self, results_dict):
        """Plot perbandingan metrics antar model"""
        self._log("Generating metrics comparison chart")
        
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=colors)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Hide unused subplot
        axes[-1].axis('off')
        plt.tight_layout()
        self._save_chart("metrics_comparison.png")
        
    def plot_training_history(self, history, model_name):
        """Plot training history untuk neural network"""
        self._log(f"Generating training history untuk {model_name}")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        filename = f"training_history_{model_name.lower().replace(' ', '_')}.png"
        self._save_chart(filename)
        
    def plot_feature_importance(self, feature_importance, top_n=15):
        """Plot feature importance"""
        self._log("Generating feature importance plot")
        
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - Gradient Boosting')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        self._save_chart("feature_importance.png")