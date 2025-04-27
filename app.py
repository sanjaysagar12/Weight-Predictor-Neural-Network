from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import logging
from ai import sigmoid  # Import your neural network functions

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model weights and normalization factors once at startup
MODEL_DIR = 'models'

# Initialize global variables
hidden_weights = None
hidden_bias = None
output_weights = None
output_bias = None
height_max = None
weight_max = None
model_loaded = False

def load_model():
    """Load the model weights and normalization factors"""
    global hidden_weights, hidden_bias, output_weights, output_bias, height_max, weight_max, model_loaded
    
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            logger.warning(f"Model directory {MODEL_DIR} not found. Creating it.")
            os.makedirs(MODEL_DIR, exist_ok=True)
            return False
            
        # Load model weights
        hidden_weights = np.load(os.path.join(MODEL_DIR, 'hidden_weights.npy'))
        hidden_bias = np.load(os.path.join(MODEL_DIR, 'hidden_bias.npy'))
        output_weights = np.load(os.path.join(MODEL_DIR, 'output_weights.npy'))
        output_bias = np.load(os.path.join(MODEL_DIR, 'output_bias.npy'))
        
        # Load normalization factors
        height_max = np.load(os.path.join(MODEL_DIR, 'height_max.npy'))
        weight_max = np.load(os.path.join(MODEL_DIR, 'weight_max.npy'))
        
        logger.info("Model loaded successfully")
        logger.info(f"Model shapes - hidden_weights: {hidden_weights.shape}, output_weights: {output_weights.shape}")
        model_loaded = True
        return True
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Try to load the model, but don't fail if it's not available yet
if not load_model():
    logger.warning("Model not found or failed to load. Please run save_model.py first.")
    # Initialize with dummy values so the app can still start
    hidden_weights = np.array([[0.5, 0.5, 0.5]])
    hidden_bias = np.array([[0.1, 0.1, 0.1]])
    output_weights = np.array([[0.5], [0.5], [0.5]])
    output_bias = np.array([[0.1]])
    height_max = 80.0
    weight_max = 250.0
    model_loaded = False

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
        
        # Make prediction using your neural network
        height_normalized = height / height_max
        
        # Log intermediate values for debugging
        logger.debug(f"Normalized height: {height_normalized}")
        
        hidden_layer_input = np.dot(height_normalized, hidden_weights) + hidden_bias
        logger.debug(f"Hidden layer input: {hidden_layer_input}")
        
        hidden_layer_output = sigmoid(hidden_layer_input)
        logger.debug(f"Hidden layer output: {hidden_layer_output}")
        
        final_input = np.dot(hidden_layer_output, output_weights) + output_bias
        logger.debug(f"Final input: {final_input}")
        
        predicted_weight = sigmoid(final_input) * weight_max
        logger.debug(f"Predicted weight (raw): {predicted_weight}")
        
        # Get the scalar value from the numpy array
        predicted_weight_value = float(predicted_weight.item(0))
        logger.info(f"Final prediction: {predicted_weight_value:.2f} pounds")
        
        return jsonify({
            'height': height,
            'predicted_weight': predicted_weight_value,
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
    return jsonify({'status': status, 'model_loaded': model_loaded})

@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    """Admin endpoint to reload the model without restarting the server"""
    success = load_model()
    return jsonify({'success': success, 'model_loaded': model_loaded})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")