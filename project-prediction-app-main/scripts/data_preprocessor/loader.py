# scripts/data_preprocessor/loader.py
import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self):
        pass
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] DataLoader: {message}")
        
    def load_all_data(self):
        """Memuat semua file CSV yang terkait dengan proyek"""
        self._log("Memulai proses loading data dari file CSV")
        
        try:
            # Load datasets utama
            self._log("Loading datasets utama...")
            allocations = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/allocations.csv')
            projects = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/projects.csv')
            team_members = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/team_members.csv')
            risks = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/risks.csv')
            project_technologies = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/project_technologies.csv')

            # Load datasets referensi
            self._log("Loading datasets referensi...")
            expertise_levels = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/expertise_levels.csv')
            sdlc_methods = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/sdlc_methods.csv')
            risk_categories = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/risk_categories.csv')
            risk_types = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/risk_types.csv')
            technology_types = pd.read_csv('/andika_projek/andika_app/project-prediction-app-main/project-prediction-app-main/data/raw/technology_types.csv')

            self._log("Load data berhasil")
            self._log(f"Jumlah data projects: {len(projects)}")
            self._log(f"Jumlah data allocations: {len(allocations)}")
            self._log(f"Jumlah data team_members: {len(team_members)}")
            self._log(f"Jumlah data risks: {len(risks)}")
            self._log(f"Jumlah data project_technologies: {len(project_technologies)}")

            return {
                'allocations': allocations,
                'projects': projects,
                'team_members': team_members,
                'risks': risks,
                'project_technologies': project_technologies,
                'expertise_levels': expertise_levels,
                'sdlc_methods': sdlc_methods,
                'risk_categories': risk_categories,
                'risk_types': risk_types,
                'technology_types': technology_types
            }

        except Exception as e:
            self._log(f"ERROR dalam loading data: {e}")
            return None