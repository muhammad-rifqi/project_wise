# scripts/feature_engineer/aggregator.py
import pandas as pd
import numpy as np
from datetime import datetime

class DataAggregator:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] DataAggregator: {message}")
        
    def aggregate_team_data(self, team_members, allocations):
        """Aggregasi data team members per project"""
        self._log("Aggregasi data team members")
        
        team_with_project = team_members.merge(allocations[['id', 'project_id']],
                                              left_on='allocation_id', right_on='id',
                                              how='left', suffixes=('', '_alloc'))

        team_agg = team_with_project.groupby('project_id').agg({
            'quantity': 'sum',
            'expertise_level_id': ['mean', 'max'],
            'avg_salary': 'mean'
        }).reset_index()

        # Flatten column names
        team_agg.columns = ['project_id', 'total_team_size', 'avg_expertise', 'max_expertise', 'avg_salary']
        
        return team_agg
        
    def aggregate_risk_data(self, risks):
        """Aggregasi data risks per project"""
        self._log("Aggregasi data risks")
        
        risk_agg = risks.groupby('project_id').agg({
            'id': 'count',
            'impact_level': lambda x: (x == 'high').sum(),
            'likelihood': lambda x: (x == 'high').sum()
        }).reset_index()
        risk_agg.columns = ['project_id', 'total_risks', 'high_impact_risks', 'high_likelihood_risks']
        
        return risk_agg
        
    def aggregate_technology_data(self, project_technologies):
        """Aggregasi data technologies per project"""
        self._log("Aggregasi data technologies")
        
        tech_agg = project_technologies.groupby('project_id').agg({
            'technology_type_id': 'nunique',
            'tool_name': 'count'
        }).reset_index()
        tech_agg.columns = ['project_id', 'unique_tech_types', 'total_tools']
        
        return tech_agg
        
    def merge_all_datasets(self, projects, allocations, team_agg, risk_agg, tech_agg):
        """Menggabungkan semua dataset"""
        self._log("Menggabungkan semua dataset")
        
        # Gabungkan projects dengan allocations
        merged_data = projects.merge(allocations, left_on='id', right_on='project_id', 
                                   how='left', suffixes=('_proj', '_alloc'))

        # Gabungkan dengan data aggregasi lainnya
        merged_data = merged_data.merge(team_agg, on='project_id', how='left')
        merged_data = merged_data.merge(risk_agg, on='project_id', how='left')
        merged_data = merged_data.merge(tech_agg, on='project_id', how='left')

        self._log(f"Shape setelah merge: {merged_data.shape}")
        
        return merged_data
        
    def handle_missing_values(self, merged_data):
        """Handling missing values"""
        self._log("Handling missing values")
        
        columns_to_fill = [
            'total_risks', 'high_impact_risks', 'high_likelihood_risks',
            'unique_tech_types', 'total_tools', 'total_team_size',
            'avg_expertise', 'max_expertise', 'avg_salary'
        ]
        
        for col in columns_to_fill:
            merged_data[col] = merged_data[col].fillna(0)
            
        self._log("Missing values handling selesai")
        
        return merged_data