# tests/test_data_preprocessor.py
import pytest
import pandas as pd
import numpy as np
from scripts.data_preprocessor.loader import DataLoader
from scripts.data_preprocessor.validator import DataValidator

class TestDataLoader:
    def test_loader_initialization(self):
        """Test inisialisasi DataLoader"""
        loader = DataLoader()
        assert loader is not None
        
    def test_load_all_data_structure(self, mocker):
        """Test struktur data yang diload"""
        # Mock file reading untuk avoid file dependencies
        mocker.patch('pandas.read_csv', return_value=pd.DataFrame({'dummy': [1]}))
        
        loader = DataLoader()
        data_dict = loader.load_all_data()
        
        # Cek bahwa method tidak error dan return dict
        assert data_dict is not None
        assert isinstance(data_dict, dict)

class TestDataValidator:
    def test_validator_initialization(self):
        """Test inisialisasi DataValidator"""
        validator = DataValidator()
        assert validator is not None
        
    def test_validate_data_quality(self, sample_data_dict):
        """Test validasi kualitas data"""
        validator = DataValidator()
        results = validator.validate_data_quality(sample_data_dict)
        
        assert results is not None
        assert isinstance(results, dict)
        assert 'projects' in results
        assert 'row_count' in results['projects']
        
    def test_check_data_consistency(self, sample_data_dict):
        """Test konsistensi data"""
        validator = DataValidator()
        results = validator.check_data_consistency(sample_data_dict)
        
        assert results is not None
        assert isinstance(results, dict)
        assert 'missing_in_allocations' in results
        assert 'extra_in_allocations' in results