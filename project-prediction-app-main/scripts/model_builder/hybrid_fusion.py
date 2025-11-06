# scripts/model_builder/hybrid_fusion.py
import numpy as np
from datetime import datetime

class HybridFeatureFusion:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] HybridFusion: {message}")
        
    def create_hybrid_features(self, gb_features, ann_embeddings):
        """Membuat hybrid features dari GB features dan ANN embeddings"""
        self._log("Membuat hybrid features")
        
        hybrid_features = np.concatenate([gb_features, ann_embeddings], axis=1)
        
        self._log(f"Hybrid features created: {hybrid_features.shape}")
        self._log(f"  - GB features: {gb_features.shape[1]}")
        self._log(f"  - ANN embeddings: {ann_embeddings.shape[1]}")
        
        return hybrid_features
        
    def create_weighted_predictions(self, gb_proba, ann_proba, nb_proba, weights=None):
        """Membuat weighted predictions dari semua model"""
        self._log("Membuat weighted predictions")
        
        if weights is None:
            # Default weights
            weights = {
                'gradient_boosting': 0.5, 
                'neural_network': 0.4, 
                'naive_bayes': 0.1
            }
        
        # Pastikan keys sesuai dengan yang digunakan
        weighted_proba = (
            weights.get('gradient_boosting', 0.4) * gb_proba +
            weights.get('neural_network', 0.4) * ann_proba +
            weights.get('naive_bayes', 0.2) * nb_proba
        )
        
        self._log(f"Weights digunakan: GB={weights.get('gradient_boosting', 0.4)}, ANN={weights.get('neural_network', 0.4)}, NB={weights.get('naive_bayes', 0.2)}")
        
        return weighted_proba