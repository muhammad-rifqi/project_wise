# scripts/model_builder/naive_bayes.py
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime

class NaiveBayesTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] NaiveBayes: {message}")
        
    def train_model(self, X_train, y_train, apply_scaling=True):
        """Training Naive Bayes model"""
        self._log("Memulai training Naive Bayes")
        
        if apply_scaling:
            self._log("Applying feature scaling untuk Naive Bayes")
            self.scaler = StandardScaler()
            X_train_processed = self.scaler.fit_transform(X_train)
        else:
            X_train_processed = X_train
            
        self.model = GaussianNB(var_smoothing=1e-2)
        self.model.fit(X_train_processed, y_train)
        
        self._log("Training Naive Bayes selesai")
        return self.model
        
    def predict(self, X_test):
        """Prediksi dengan Naive Bayes"""
        if self.scaler is not None:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test
            
        return self.model.predict(X_test_processed), self.model.predict_proba(X_test_processed)