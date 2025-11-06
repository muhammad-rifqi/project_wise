# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
# HAPUS import make_classification dari sini karena sudah di test_integration.py

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_data_dict():
    """Fixture untuk sample data dictionary"""
    return {
        'projects': pd.DataFrame({
            'id': [1, 2, 3],
            'project_name': ['Project A', 'Project B', 'Project C'],
            'duration_months': [6, 12, 8],
            'total_development_cost': [50000, 100000, 75000],
            'additional_costs': [5000, 10000, 7500],
            'scale': ['medium', 'high', 'low'],
            'type_project': ['web', 'mobile', 'desktop'],
            'sdlc_method_id': [1, 2, 1],
            'status_project': ['success', 'failed', 'success'],
            'start_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'end_date': ['2023-07-01', '2024-02-01', '2023-11-01']
        }),
        'allocations': pd.DataFrame({
            'id': [1, 2, 3],
            'project_id': [1, 2, 3]
        }),
        'team_members': pd.DataFrame({
            'allocation_id': [1, 1, 2, 2, 3],
            'quantity': [2, 3, 1, 2, 4],
            'expertise_level_id': [5, 6, 4, 5, 7],
            'avg_salary': [5000, 6000, 4500, 5500, 7000]
        }),
        'risks': pd.DataFrame({
            'id': [1, 2, 3, 4],
            'project_id': [1, 1, 2, 3],
            'impact_level': ['high', 'medium', 'low', 'high'],
            'likelihood': ['medium', 'high', 'low', 'medium']
        }),
        'project_technologies': pd.DataFrame({
            'project_id': [1, 1, 2, 3],
            'technology_type_id': [1, 2, 1, 3],
            'tool_name': ['React', 'Node.js', 'Flutter', 'Django']
        })
    }

@pytest.fixture
def sample_model_data():
    """Fixture untuk sample model data"""
    return pd.DataFrame({
        'duration_months': [6, 12, 8, 10],
        'total_development_cost': [50000, 100000, 75000, 90000],
        'additional_costs': [5000, 10000, 7500, 8000],
        'total_team_size': [5, 3, 4, 6],
        'avg_expertise': [5.5, 4.0, 7.0, 6.0],
        'max_expertise': [6, 4, 7, 7],
        'avg_salary': [5500, 4500, 7000, 6000],
        'total_risks': [2, 1, 1, 3],
        'high_impact_risks': [1, 0, 0, 2],
        'high_likelihood_risks': [1, 0, 1, 1],
        'unique_tech_types': [2, 1, 1, 2],
        'total_tools': [2, 1, 1, 2],
        'scale': ['medium', 'high', 'low', 'medium'],
        'type_project': ['web', 'mobile', 'desktop', 'web'],
        'sdlc_method_id': [1, 2, 1, 2],
        'project_success': [1, 0, 1, 0]
    })

@pytest.fixture
def sample_training_data():
    """Fixture untuk sample training data"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def sample_feature_importance():
    """Fixture untuk sample feature importance"""
    return pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(10)],
        'importance': np.random.rand(10)
    }).sort_values('importance', ascending=False)