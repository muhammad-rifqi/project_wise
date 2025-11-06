# scripts/feature_engineer/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataVisualizer:
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
        print(f"[{current_time}] DataVisualizer: {message}")
        
    def _save_chart(self, filename):
        """Menyimpan chart dengan penomoran otomatis"""
        chart_number = str(self.chart_counter).zfill(2)
        full_filename = f"{chart_number}_{filename}"
        full_path = os.path.join(self.output_dir, full_filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        self._log(f"Chart disimpan: {full_filename}")
        self.chart_counter += 1
        plt.close()  # Tutup plot untuk menghemat memory
        
    def plot_target_distribution(self, model_data):
        """Plot distribusi target variable"""
        self._log("Generating target distribution plot")
        
        plt.figure(figsize=(8, 6))
        model_data['project_success'].value_counts().plot(kind='bar', color=['red', 'green'])
        plt.title('Distribusi Target Variable')
        plt.xlabel('Project Success')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Failed', 'Success'], rotation=0)
        
        self._save_chart("target_distribution.png")
        
    def plot_feature_distributions(self, model_data, numerical_features):
        """Plot distribusi features numerik"""
        self._log("Generating feature distribution plots")
        
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.hist(model_data[feature], bins=20, alpha=0.7, color='blue')
            plt.title(f'Distribusi {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            
        plt.tight_layout()
        self._save_chart("feature_distributions.png")
        
    def plot_correlation_matrix(self, model_data, numerical_features):
        """Plot correlation matrix untuk features numerik"""
        self._log("Generating correlation matrix")
        
        # Ambil subset features untuk correlation matrix
        numeric_features_for_corr = numerical_features[:8]
        
        correlation_matrix = model_data[numeric_features_for_corr].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix Features Numerik')
        plt.tight_layout()
        self._save_chart("correlation_matrix.png")