# app/model_loader.py
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

class ProjectPredictor:
    def __init__(self):
        self.scaler = None
        self.gb_model = None
        self.ann_model = None
        self.embedding_model = None
        self.nb_model = None
        self.feature_names = None
        self.is_loaded = False
    
    def load_models(self):
        """Load semua trained models dari folder models/"""
        try:
            print("📥 Loading trained models...")
            
            # Load scaler
            if os.path.exists('../models/scaler.pkl'):
                self.scaler = joblib.load('../models/scaler.pkl')
                print("Scaler loaded")
            
            # Load Gradient Boosting
            if os.path.exists('../models/gradient_boosting_model.pkl'):
                self.gb_model = joblib.load('../models/gradient_boosting_model.pkl')
                print("Gradient Boosting model loaded")
            
            # Load ANN model
            if os.path.exists('../models/ann_model.h5'):
                self.ann_model = load_model('../models/ann_model.h5')
                # Buat embedding model dari ANN
                self.embedding_model = Model(
                    inputs=self.ann_model.input,
                    outputs=self.ann_model.get_layer('embedding_layer').output
                )
                print("ANN model loaded")
            
            # Load Naive Bayes
            if os.path.exists('../models/naive_bayes_model.pkl'):
                self.nb_model = joblib.load('../models/naive_bayes_model.pkl')
                print("Naive Bayes model loaded")
            
            # Load feature names (jika ada)
            if os.path.exists('../models/feature_info.pkl'):
                feature_info = joblib.load('../models/feature_info.pkl')
                self.feature_names = feature_info.get('feature_names', [])
                print("Feature info loaded")
            
            self.is_loaded = True
            print("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data seperti di training - PERSIS dari notebook"""
        try:
            # Convert input ke DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Feature engineering seperti di notebook
            # Scale mapping - PERSIS dari notebook
            scale_mapping = {'low': 0, 'medium': 1, 'high': 2}
            if 'scale' in input_df.columns:
                input_df['scale_encoded'] = input_df['scale'].map(scale_mapping)
            
            # One-hot encoding untuk type_project - PERSIS dari notebook
            if 'type_project' in input_df.columns:
                type_dummies = pd.get_dummies(input_df['type_project'], prefix='type')
                # Ensure all expected type columns exist
                expected_types = ['type_web_development', 'type_mobile_app', 'type_data_analytics', 
                                'type_cloud_migration', 'type_ai_ml', 'type_iot', 'type_blockchain']
                for expected_col in expected_types:
                    if expected_col not in type_dummies.columns:
                        type_dummies[expected_col] = 0
                input_df = pd.concat([input_df, type_dummies], axis=1)
            
            # Select final features seperti di training - PERSIS dari notebook
            numerical_features = [
                'duration_months', 'total_development_cost', 'additional_costs',
                'total_team_size', 'avg_expertise', 'max_expertise', 'avg_salary',
                'total_risks', 'high_impact_risks', 'high_likelihood_risks',
                'unique_tech_types', 'total_tools', 'sdlc_method_id', 'scale_encoded'
            ]
            
            # Add type_project dummies
            type_cols = [col for col in input_df.columns if col.startswith('type_')]
            final_features = numerical_features + type_cols
            
            # Hanya ambil columns yang ada
            available_features = [f for f in final_features if f in input_df.columns]
            processed_data = input_df[available_features]
            
            # Fill missing values dengan 0
            processed_data = processed_data.fillna(0)
            
            # Scale features
            if self.scaler:
                scaled_data = self.scaler.transform(processed_data)
            else:
                scaled_data = processed_data.values
            
            return scaled_data, processed_data
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            return None, None
    
    def predict(self, input_data):
        """Prediction menggunakan hybrid model - PERSIS dari notebook flow"""
        if not self.is_loaded:
            return {"error": "Models not loaded. Please run training first."}
        
        try:
            # Preprocess input
            scaled_data, processed_data = self.preprocess_input(input_data)
            
            if scaled_data is None:
                return {"error": "Preprocessing failed"}
            
            # Feature selection menggunakan Gradient Boosting - PERSIS dari notebook
            from sklearn.feature_selection import SelectFromModel
            selector = SelectFromModel(self.gb_model, prefit=True, threshold='median')
            gb_selected_features = selector.transform(scaled_data)
            
            # ANN embedding - PERSIS dari notebook
            ann_embedding = self.embedding_model.predict(scaled_data)
            
            # Hybrid features - PERSIS dari notebook
            hybrid_features = np.concatenate([gb_selected_features, ann_embedding], axis=1)
            
            # Naive Bayes prediction - PERSIS dari notebook
            nb_prediction = self.nb_model.predict(hybrid_features)
            nb_probability = self.nb_model.predict_proba(hybrid_features)
            
            # Juga dapatkan prediction dari individual models
            gb_prediction = self.gb_model.predict(scaled_data)
            gb_probability = self.gb_model.predict_proba(scaled_data)
            
            ann_prediction = (self.ann_model.predict(scaled_data) > 0.5).astype(int).flatten()
            ann_probability = self.ann_model.predict(scaled_data).flatten()
            
            # Return results - PERSIS dari notebook structure
            return {
                'hybrid': {
                    'prediction': int(nb_prediction[0]),
                    'probability': float(nb_probability[0][1]),
                    'confidence': 'High' if nb_probability[0][1] > 0.7 else 'Medium' if nb_probability[0][1] > 0.6 else 'Low'
                },
                'gradient_boosting': {
                    'prediction': int(gb_prediction[0]),
                    'probability': float(gb_probability[0][1])
                },
                'ann': {
                    'prediction': int(ann_prediction[0]),
                    'probability': float(ann_probability[0])
                },
                'success': bool(nb_prediction[0]),
                'confidence_score': float(nb_probability[0][1]),
                'message': 'Hybrid model prediction completed successfully'
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}

# Global instance
predictor = ProjectPredictor()