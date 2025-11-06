# scripts/data_preprocessor/validator.py
import pandas as pd
import numpy as np
from datetime import datetime

class DataValidator:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] DataValidator: {message}")
        
    def validate_data_quality(self, data_dict):
        """Validasi kualitas data untuk setiap dataset"""
        self._log("Memulai validasi kualitas data")
        
        validation_results = {}
        
        for name, df in data_dict.items():
            self._log(f"Validasi dataset: {name}")
            
            validation_results[name] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Tampilkan summary untuk setiap dataset
            self._log(f"  - Rows: {len(df)}, Columns: {len(df.columns)}")
            self._log(f"  - Missing values: {df.isnull().sum().sum()}")
            self._log(f"  - Duplicate rows: {df.duplicated().sum()}")
        
        return validation_results
        
    def check_data_consistency(self, data_dict):
        """Cek konsistensi data antar dataset"""
        self._log("Memeriksa konsistensi data antar dataset")
        
        # Cek konsistensi ID antara projects dan allocations
        projects_ids = set(data_dict['projects']['id'])
        allocations_project_ids = set(data_dict['allocations']['project_id'])
        
        missing_in_allocations = projects_ids - allocations_project_ids
        extra_in_allocations = allocations_project_ids - projects_ids
        
        self._log(f"Projects tanpa allocations: {len(missing_in_allocations)}")
        self._log(f"Allocations tanpa projects: {len(extra_in_allocations)}")
        
        return {
            'missing_in_allocations': missing_in_allocations,
            'extra_in_allocations': extra_in_allocations
        }