import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import mlflow
import logging
from datetime import datetime
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
model_metadata = {}

# HTML template for the prediction form
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prostate Cancer Risk Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .result.low { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .result.medium { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .result.high { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prostate Cancer Risk Prediction</h1>
        <p>Enter patient information to predict prostate cancer risk level:</p>
        
        <form method="POST">
            <div class="form-group">
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" min="18" max="100" required>
            </div>
            
            <div class="form-group">
                <label for="bmi">BMI:</label>
                <input type="number" id="bmi" name="bmi" step="0.1" min="15" max="50" required>
            </div>
            
            <div class="form-group">
                <label for="sleep_hours">Sleep Hours:</label>
                <input type="number" id="sleep_hours" name="sleep_hours" step="0.1" min="3" max="12" required>
            </div>
            
            <div class="form-group">
                <label for="family_history">Family History:</label>
                <select id="family_history" name="family_history" required>
                    <option value="">Select...</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="diet">Diet:</label>
                <select id="diet" name="diet" required>
                    <option value="">Select...</option>
                    <option value="Healthy">Healthy</option>
                    <option value="Moderate">Moderate</option>
                    <option value="Unhealthy">Unhealthy</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="exercise_frequency">Exercise Frequency:</label>
                <select id="exercise_frequency" name="exercise_frequency" required>
                    <option value="">Select...</option>
                    <option value="Daily">Daily</option>
                    <option value="Weekly">Weekly</option>
                    <option value="Monthly">Monthly</option>
                    <option value="Rarely">Rarely</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="smoking_status">Smoking Status:</label>
                <select id="smoking_status" name="smoking_status" required>
                    <option value="">Select...</option>
                    <option value="Never">Never</option>
                    <option value="Former">Former</option>
                    <option value="Current">Current</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="alcohol_consumption">Alcohol Consumption:</label>
                <select id="alcohol_consumption" name="alcohol_consumption" required>
                    <option value="">Select...</option>
                    <option value="None">None</option>
                    <option value="Light">Light</option>
                    <option value="Moderate">Moderate</option>
                    <option value="Heavy">Heavy</option>
                </select>
            </div>
            
            <button type="submit">Predict Risk</button>
        </form>
        
        {% if prediction %}
        <div class="result {{ prediction.risk_class }}">
            <h3>Prediction Result</h3>
            <p><strong>Risk Level:</strong> {{ prediction.risk_level }}</p>
            <p><strong>Confidence:</strong> {{ "%.2f"|format(prediction.confidence * 100) }}%</p>
            <p><strong>Probabilities:</strong></p>
            <ul>
                <li>Low Risk: {{ "%.2f"|format(prediction.probabilities.low * 100) }}%</li>
                <li>Medium Risk: {{ "%.2f"|format(prediction.probabilities.medium * 100) }}%</li>
                <li>High Risk: {{ "%.2f"|format(prediction.probabilities.high * 100) }}%</li>
            </ul>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

def load_model():
    """Load the trained model"""
    global model, model_metadata
    
    try:
        # Try to load from MLflow first
        model_path = "models/model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
            
            # Load run ID if available
            run_id_path = "models/run_id/run_id.txt"
            if os.path.exists(run_id_path):
                with open(run_id_path, 'r') as f:
                    run_id = f.read().strip()
                model_metadata['run_id'] = run_id
                logger.info(f"Model run ID: {run_id}")
        else:
            logger.error("Model file not found. Please train a model first.")
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
    
    return True

def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Define categorical columns and their expected values
    categorical_mappings = {
        'family_history': {'No': 0, 'Yes': 1},
        'diet': {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2},
        'exercise_frequency': {'Daily': 0, 'Monthly': 1, 'Rarely': 2, 'Weekly': 3},
        'smoking_status': {'Current': 0, 'Former': 1, 'Never': 2},
        'alcohol_consumption': {'Heavy': 0, 'Light': 1, 'Moderate': 2, 'None': 3}
    }
    
    # Apply categorical mappings
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

def save_prediction_log(input_data, prediction, probabilities):
    """Save prediction for monitoring purposes"""
    try:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'model_run_id': model_metadata.get('run_id', 'unknown')
        }
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Append to predictions log file
        log_file = logs_dir / "predictions.jsonl"
        with open(log_file, 'a') as f:
            import json
            f.write(json.dumps(log_data) + '\n')
            
    except Exception as e:
        logger.error(f"Error saving prediction log: {e}")

@app.route('/', methods=['GET', 'POST'])
def predict():
    """Main prediction endpoint"""
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)
    
    try:
        # Get form data
        input_data = {
            'age': float(request.form['age']),
            'bmi': float(request.form['bmi']),
            'sleep_hours': float(request.form['sleep_hours']),
            'family_history': request.form['family_history'],
            'diet': request.form['diet'],
            'exercise_frequency': request.form['exercise_frequency'],
            'smoking_status': request.form['smoking_status'],
            'alcohol_consumption': request.form['alcohol_consumption']
        }
        
        # Preprocess input
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        # Map prediction to risk level
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_level = risk_levels.get(prediction, 'Unknown')
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        # Save prediction log
        save_prediction_log(input_data, prediction, probabilities)
        
        # Prepare result
        result = {
            'risk_level': risk_level,
            'risk_class': risk_level.lower(),
            'confidence': confidence,
            'probabilities': {
                'low': probabilities[0],
                'medium': probabilities[1],
                'high': probabilities[2]
            }
        }
        
        return render_template_string(HTML_TEMPLATE, prediction=result)
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return render_template_string(HTML_TEMPLATE, 
                                    error=f"Error making prediction: {str(e)}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint for predictions"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        # Map prediction to risk level
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_level = risk_levels.get(prediction, 'Unknown')
        
        # Save prediction log
        save_prediction_log(data, prediction, probabilities)
        
        # Return result
        result = {
            'risk_level': risk_level,
            'prediction': int(prediction),
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'confidence': float(max(probabilities)),
            'model_run_id': model_metadata.get('run_id', 'unknown')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_run_id': model_metadata.get('run_id', 'unknown')
    })

@app.route('/model_info')
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model_type': 'XGBoost Classifier',
        'model_run_id': model_metadata.get('run_id', 'unknown'),
        'features': [
            'age', 'bmi', 'sleep_hours', 'family_history', 
            'diet', 'exercise_frequency', 'smoking_status', 'alcohol_consumption'
        ],
        'target_classes': ['Low', 'Medium', 'High']
    })

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=9696, debug=False)