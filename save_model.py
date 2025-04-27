import numpy as np
import os
import sys
import logging

from ai import NeuralNetwork

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_model():
    """Train and save a neural network model"""
    try:
        # Create a new neural network
        logger.info("Creating a new neural network model...")
        nn = NeuralNetwork(input_neurons=1, hidden_neurons=3, output_neurons=1)
        
        # Load dataset
        logger.info("Loading dataset...")
        try:
            nn.load_data('height_weight.csv')
        except FileNotFoundError:
            logger.error("Dataset file 'height_weight.csv' not found.")
            return False
            
        # Train the model
        logger.info("Training the model...")
        nn.train(epochs=1000)
        
        # Save the model
        logger.info("Saving the trained model...")
        nn.save_weights('models')
        
        return True
    except Exception as e:
        logger.error(f"Error while training and saving the model: {e}")
        return False

def load_existing_model():
    """Load an existing neural network model to verify it works"""
    try:
        # Create a new neural network
        logger.info("Creating a neural network instance...")
        nn = NeuralNetwork()
        
        # Load weights
        logger.info("Loading model weights...")
        nn.load_weights('models')
        
        # Test prediction
        sample_height = 70  # inches
        predicted_weight = nn.predict(sample_height)
        logger.info(f"Test prediction - Height: {sample_height} inches, Weight: {predicted_weight:.2f} pounds")
        
        return True
    except Exception as e:
        logger.error(f"Error while loading the model: {e}")
        return False

if __name__ == "__main__":
    print("Neural Network Model Manager")
    print("===========================")
    print("1. Train and save a new model")
    print("2. Load and test existing model")
    print("3. Both (train new and then test)")
    print("===========================")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        if save_model():
            print("Model trained and saved successfully!")
        else:
            print("Failed to train and save the model.")
            sys.exit(1)
    elif choice == '2':
        if load_existing_model():
            print("Model loaded and tested successfully!")
        else:
            print("Failed to load the model. You may need to train a new model first.")
            sys.exit(1)
    elif choice == '3':
        if save_model() and load_existing_model():
            print("Model trained, saved, and tested successfully!")
        else:
            print("Error occurred during model training or testing.")
            sys.exit(1)
    else:
        print("Invalid choice!")
        sys.exit(1)