# scripts/model_evaluator/comparator.py
import pandas as pd
from datetime import datetime

class ModelComparator:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] ModelComparator: {message}")
        
    def create_comparison_table(self, individual_results, hybrid_results):
        """Membuat tabel perbandingan performa model"""
        self._log("Membuat comparison table")
        
        comparison_data = []
        
        # Add individual models
        for model_name, results in individual_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'ROC-AUC': results['roc_auc']
            })
        
        # Add hybrid model
        comparison_data.append({
            'Model': 'Hybrid Weighted Voting',
            'Accuracy': hybrid_results['accuracy'],
            'Precision': hybrid_results['precision'],
            'Recall': hybrid_results['recall'],
            'F1-Score': hybrid_results['f1'],
            'ROC-AUC': hybrid_results['roc_auc']
        })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
        
    def print_detailed_comparison(self, individual_results, hybrid_results):
        """Print detailed comparison analysis"""
        self._log("Melakukan analisis komparatif detail")
        
        print("\n" + "="*80)
        print("COMPARATIVE MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Table header
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        # Individual models
        for model_name, results in individual_results.items():
            print(f"{model_name.replace('_', ' ').title():<25} "
                  f"{results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                  f"{results['recall']:<10.4f} {results['f1']:<10.4f} {results['roc_auc']:<10.4f}")
        
        # Hybrid model
        print(f"{'Hybrid Weighted Voting':<25} "
              f"{hybrid_results['accuracy']:<10.4f} {hybrid_results['precision']:<10.4f} "
              f"{hybrid_results['recall']:<10.4f} {hybrid_results['f1']:<10.4f} {hybrid_results['roc_auc']:<10.4f}")
        
        print("=" * 80)
        
        # Best model identification
        all_results = list(individual_results.items()) + [('Hybrid', hybrid_results)]
        
        best_f1_model = max(all_results, key=lambda x: x[1]['f1'])
        best_auc_model = max(all_results, key=lambda x: x[1]['roc_auc'])
        best_accuracy_model = max(all_results, key=lambda x: x[1]['accuracy'])
        
        print(f"\nBEST PERFORMING MODELS:")
        print(f"Best F1-Score: {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")
        print(f"Best ROC-AUC:  {best_auc_model[0]} ({best_auc_model[1]['roc_auc']:.4f})")
        print(f"Best Accuracy: {best_accuracy_model[0]} ({best_accuracy_model[1]['accuracy']:.4f})")
        
        # Improvement analysis
        best_individual_f1 = max([results['f1'] for _, results in individual_results.items()])
        hybrid_f1 = hybrid_results['f1']
        improvement = ((hybrid_f1 - best_individual_f1) / best_individual_f1) * 100
        
        print(f"\nHybrid vs Best Individual Improvement: {improvement:+.2f}%")