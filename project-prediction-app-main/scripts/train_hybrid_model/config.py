# scripts/train_hybrid_model/config.py
class ModelConfig:
    """Configuration class for hybrid model training"""
    
    # Random seed
    RANDOM_STATE = 42
    
    # Training parameters
    TEST_SIZE = 0.2
    ANN_EPOCHS = 300
    ANN_BATCH_SIZE = 16
    ANN_PATIENCE = 25
    
    # Model weights for hybrid - gunakan keys yang konsisten
    MODEL_WEIGHTS = {
        'gradient_boosting': 0.5,
        'neural_network': 0.4, 
        'naive_bayes': 0.1
    }
    
    # Feature selection
    FEATURE_SELECTION_THRESHOLD = 'median'
    
    # Naive Bayes
    NAIVE_BAYES_VAR_SMOOTHING = 1e-2