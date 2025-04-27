import numpy as np
import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_model_from_ai():
    """Save model weights from the ai.py file"""
    try:
        # Import neural network model from ai.py
        from ai import (
            hidden_weights,
            hidden_bias,
            output_weights, 
            output_bias,
            data
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the trained model weights
        np.save('models/hidden_weights.npy', hidden_weights)
        np.save('models/hidden_bias.npy', hidden_bias)
        np.save('models/output_weights.npy', output_weights)
        np.save('models/output_bias.npy', output_bias)
        
        # Save normalization factors
        height_max = np.max(data['Height(Inches)'])
        weight_max = np.max(data['Weight(Pounds)'])
        
        np.save('models/height_max.npy', height_max)
        np.save('models/weight_max.npy', weight_max)
        
        logger.info("Model weights and normalization factors saved to 'models/' directory")
        logger.info(f"Model shapes - hidden_weights: {hidden_weights.shape}, output_weights: {output_weights.shape}")
        logger.info(f"Normalization factors - height_max: {height_max}, weight_max: {weight_max}")
        return True
    except ImportError:
        logger.error("Failed to import variables from ai.py. Make sure the model is trained.")
        return False
    except AttributeError as e:
        logger.error(f"AttributeError: {e}. Make sure the model variables exist in ai.py.")
        return False
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False

def train_and_save_model():
    """Train a new model and save it"""
    try:
        # Import the data
        data = pd.read_csv('height_weight.csv')
        X = data['Height(Inches)'].values.reshape(-1, 1)
        Y = data['Weight(Pounds)'].values.reshape(-1, 1)
        
        # Normalize
        X = X / np.max(X)
        Y = Y / np.max(Y)
        
        # Define sigmoid function
        def sigmoid(x):
            x_safe = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x_safe))
            
        def sigmoid_derivative(x):
            return x * (1 - x)
        
        # Initialize parameters
        np.random.seed(42)
        input_neurons = 1
        hidden_neurons = 3
        output_neurons = 1
        
        # Random initial weights and biases
        hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
        hidden_bias = np.random.uniform(size=(1, hidden_neurons))
        output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
        output_bias = np.random.uniform(size=(1, output_neurons))
        
        # Training parameters
        learning_rate = 0.01
        epochs = 1000
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
            hidden_layer_output = sigmoid(hidden_layer_input)
            
            final_input = np.dot(hidden_layer_output, output_weights) + output_bias
            final_output = sigmoid(final_input)
            
            # Loss calculation
            loss = np.mean((Y - final_output) ** 2)
            
            if epoch % 200 == 0:
                logger.info(f"Training Epoch {epoch}, Loss: {loss:.6f}")
                
            # Backward pass
            error = Y - final_output
            d_output = error * sigmoid_derivative(final_output)
            
            error_hidden_layer = d_output.dot(output_weights.T)
            d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
            
            # Update weights
            output_weights += hidden_layer_output.T.dot(d_output) * learning_rate
            output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            
            hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
            hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
        
        logger.info(f"Training complete. Final loss: {loss:.6f}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the trained model weights
        np.save('models/hidden_weights.npy', hidden_weights)
        np.save('models/hidden_bias.npy', hidden_bias)
        np.save('models/output_weights.npy', output_weights)
        np.save('models/output_bias.npy', output_bias)
        
        # Save normalization factors
        height_max = np.max(data['Height(Inches)'])
        weight_max = np.max(data['Weight(Pounds)'])
        
        np.save('models/height_max.npy', height_max)
        np.save('models/weight_max.npy', weight_max)
        
        logger.info("Model weights and normalization factors saved to 'models/' directory")
        logger.info(f"Model shapes - hidden_weights: {hidden_weights.shape}, output_weights: {output_weights.shape}")
        logger.info(f"Normalization factors - height_max: {height_max}, weight_max: {weight_max}")
        return True
    except Exception as e:
        logger.error(f"Failed to train and save model: {e}")
        return False

if __name__ == "__main__":
    print("Saving neural network model...")
    
    # First try to save from existing ai.py
    if not save_model_from_ai():
        print("\nFailed to save model from ai.py. Would you like to train a new model? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            print("Training new model...")
            if train_and_save_model():
                print("New model trained and saved successfully!")
            else:
                print("Failed to train and save new model.")
                sys.exit(1)
        else:
            print("Model saving canceled.")
            sys.exit(1)
    
    print("Model saved successfully!")