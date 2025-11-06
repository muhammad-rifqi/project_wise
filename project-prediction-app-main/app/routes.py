# app/routes.py
from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np

main = Blueprint('main', __name__)

# Options untuk dropdown - SESUAI dengan data Anda
PROJECT_TYPES = [
    'web_development', 'mobile_app', 'data_analytics', 
    'cloud_migration', 'ai_ml', 'iot', 'blockchain'
]

SDLC_METHODS = {
    1: 'Waterfall',
    2: 'Agile', 
    3: 'Scrum',
    4: 'Kanban', 
    5: 'DevOps'
}

@main.route('/')
def index():
    return render_template('index.html', 
                         project_types=PROJECT_TYPES,
                         sdlc_methods=SDLC_METHODS)

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data - SESUAI dengan features di notebook
        form_data = {
            'duration_months': float(request.form.get('duration_months', 0)),
            'total_development_cost': float(request.form.get('total_development_cost', 0)),
            'additional_costs': float(request.form.get('additional_costs', 0)),
            'total_team_size': int(request.form.get('total_team_size', 0)),
            'avg_expertise': float(request.form.get('avg_expertise', 0)),
            'max_expertise': float(request.form.get('max_expertise', 0)),
            'avg_salary': float(request.form.get('avg_salary', 0)),
            'total_risks': int(request.form.get('total_risks', 0)),
            'high_impact_risks': int(request.form.get('high_impact_risks', 0)),
            'high_likelihood_risks': int(request.form.get('high_likelihood_risks', 0)),
            'unique_tech_types': int(request.form.get('unique_tech_types', 0)),
            'total_tools': int(request.form.get('total_tools', 0)),
            'sdlc_method_id': int(request.form.get('sdlc_method_id', 0)),
            'scale': request.form.get('scale', 'medium'),
            'type_project': request.form.get('type_project', 'web_development')
        }
        
        # Lakukan prediksi
        from app.model_loader import predictor
        prediction_result = predictor.predict(form_data)
        
        if 'error' in prediction_result:
            return render_template('error.html', error=prediction_result['error'])
        
        return render_template('results.html', 
                             prediction=prediction_result,
                             form_data=form_data)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@main.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Validasi required fields
        required_fields = ['duration_months', 'total_development_cost', 'total_team_size']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Lakukan prediksi
        from app.model_loader import predictor
        prediction_result = predictor.predict(data)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/api/health')
def api_health():
    return jsonify({"status": "healthy", "message": "Project Prediction API is working"})

@main.route('/api/project_types')
def api_project_types():
    return jsonify(PROJECT_TYPES)

@main.route('/api/sdlc_methods') 
def api_sdlc_methods():
    return jsonify(SDLC_METHODS)