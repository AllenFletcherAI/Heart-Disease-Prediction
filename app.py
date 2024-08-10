from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from flask_talisman import Talisman
import joblib
import pandas as pd
import logging
import os
from werkzeug.exceptions import BadRequest

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)
Talisman(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Random Forest model
model_path = os.getenv('MODEL_PATH', 'models/random_forest_model.joblib')
try:
    random_forest_model = joblib.load(model_path)
    logging.info(f"Model successfully loaded from {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Define a function for preprocessing incoming data
def preprocess_data(data):
    try:
        # Validate input data
        if not isinstance(data, dict):
            raise BadRequest("Invalid input format")
        
        # Convert incoming data to a DataFrame
        df = pd.DataFrame([data])
        
        # Example preprocessing steps
        # df.fillna(0, inplace=True)
        # Add more preprocessing steps as needed

        return df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        if not data:
            raise ValueError("No data provided")
        
        # Preprocess the data
        processed_data = preprocess_data(data)
        
        # Perform prediction with Random Forest model
        prediction = random_forest_model.predict(processed_data)
        
        # Prepare the response
        response = {
            'prediction': prediction.tolist()
        }
        
        # Create a response with no cache headers
        resp = make_response(jsonify(response))
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        
        return resp, 200
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({'status': 'healthy'}), 200
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy'}), 500

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    debug_mode = os.getenv('DEBUG_MODE', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)

