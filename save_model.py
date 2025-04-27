import numpy as np
import os
import sys
import logging
import argparse

from ai import NeuralNetwork

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_model(epochs=1000, hidden_neurons=3, learning_rate=0.01, data_file='height_weight.csv'):
    """Train and save a neural network model"""
    try:
        # Create a new neural network
        logger.info(f"Creating a new neural network model with {hidden_neurons} hidden neurons and {learning_rate} learning rate...")
        nn = NeuralNetwork(input_neurons=1, hidden_neurons=hidden_neurons, output_neurons=1, learning_rate=learning_rate)
        
        # Load dataset
        logger.info(f"Loading dataset from {data_file}...")
        try:
            nn.load_data(data_file)
        except FileNotFoundError:
            logger.error(f"Dataset file '{data_file}' not found.")
            return False
            
        # Train the model
        logger.info(f"Training the model for {epochs} epochs...")
        nn.train(epochs=epochs)
        
        # Save the model
        logger.info("Saving the trained model...")
        nn.save_weights('models')
        
        return True
    except Exception as e:
        logger.error(f"Error while training and saving the model: {e}")
        return False

def load_existing_model(data_file='height_weight.csv'):
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Model Manager')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train and save a new model')
    train_parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    train_parser.add_argument('--hidden', type=int, default=3, help='Number of hidden neurons')
    train_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    train_parser.add_argument('--data', type=str, default='height_weight.csv', help='Path to data file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Load and test existing model')
    test_parser.add_argument('--data', type=str, default='height_weight.csv', help='Path to data file')
    
    # Both command
    both_parser = subparsers.add_parser('both', help='Both train new and then test')
    both_parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    both_parser.add_argument('--hidden', type=int, default=3, help='Number of hidden neurons')
    both_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    both_parser.add_argument('--data', type=str, default='height_weight.csv', help='Path to data file')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Default to menu if no arguments provided
    if not args.command:
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
    else:
        # Process command-line arguments
        if args.command == 'train':
            if save_model(epochs=args.epochs, hidden_neurons=args.hidden, 
                         learning_rate=args.lr, data_file=args.data):
                print("Model trained and saved successfully!")
            else:
                print("Failed to train and save the model.")
                sys.exit(1)
        elif args.command == 'test':
            if load_existing_model(data_file=args.data):
                print("Model loaded and tested successfully!")
            else:
                print("Failed to load the model. You may need to train a new model first.")
                sys.exit(1)
        elif args.command == 'both':
            if save_model(epochs=args.epochs, hidden_neurons=args.hidden, 
                         learning_rate=args.lr, data_file=args.data) and load_existing_model(data_file=args.data):
                print("Model trained, saved, and tested successfully!")
            else:
                print("Error occurred during model training or testing.")
                sys.exit(1)