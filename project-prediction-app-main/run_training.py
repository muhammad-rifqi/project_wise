# run_training.py
"""
Main entry point for running the hybrid model training pipeline
"""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.train_hybrid_model.trainer import main

if __name__ == "__main__":
    main()