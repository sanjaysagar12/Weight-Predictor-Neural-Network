from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import logging
from ai import NeuralNetwork

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the neural network
nn = NeuralNetwork()
model_loaded = False

def load_model():
    """Load the model weights"""
    global nn, model_loaded
    
    try:
        logger.info("Loading neural network model...")
        nn.load_weights('models')
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

# Try to load the model on startup
try:
    load_model()
except:
    logger.warning("Model not found. Please run save_model.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_loaded:
        logger.warning("Prediction attempted but model is not loaded")
        return jsonify({
            'error': 'Model not loaded. Please run save_model.py first.',
            'success': False
        }), 400
        
    try:
        # Log the incoming request data
        data = request.get_json()
        logger.info(f"Prediction request with data: {data}")
        
        if not data or 'height' not in data:
            logger.warning("Missing height parameter in request")
            return jsonify({
                'error': 'Missing height parameter',
                'success': False
            }), 400
            
        height = float(data['height'])
        logger.info(f"Making prediction for height: {height}")
        
        # Make prediction using the neural network
        predicted_weight = nn.predict(height)
        logger.info(f"Final prediction: {predicted_weight:.2f} pounds")
        
        return jsonify({
            'height': height,
            'predicted_weight': predicted_weight,
            'success': True
        })
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': f"Invalid input: {str(e)}",
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': f"Prediction error: {str(e)}",
            'success': False
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'ok' if model_loaded else 'model_not_loaded'
    return jsonify({
        'status': status, 
        'model_loaded': model_loaded
    })

@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    """Admin endpoint to reload the model without restarting the server"""
    success = load_model()
    return jsonify({
        'success': success, 
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    app.run(debug=True)