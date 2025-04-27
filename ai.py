import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_neurons=1, hidden_neurons=3, output_neurons=1, learning_rate=0.01, seed=42):
        """
        Initialize the Neural Network with the given architecture
        
        Args:
            input_neurons (int): Number of input neurons
            hidden_neurons (int): Number of hidden neurons
            output_neurons (int): Number of output neurons
            learning_rate (float): Learning rate for gradient descent
            seed (int): Random seed for reproducibility
        """
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Initialize weights and biases
        self.hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
        self.hidden_bias = np.random.uniform(size=(1, hidden_neurons))
        self.output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
        self.output_bias = np.random.uniform(size=(1, output_neurons))
        
        # For storing normalization factors
        self.X_max = None
        self.Y_max = None
        
        # For storing the dataset
        self.data = None

    def sigmoid(self, x):
        """Sigmoid activation function with clipping for numerical stability"""
        x_safe = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_safe))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def load_data(self, filename):
        """
        Load data from a CSV file
        
        Args:
            filename (str): Path to the CSV file
        """
        self.data = pd.read_csv(filename)
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the data by reshaping and normalizing
        
        Returns:
            tuple: Normalized X and Y data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        X = self.data['Height(Inches)'].values.reshape(-1, 1)
        Y = self.data['Weight(Pounds)'].values.reshape(-1, 1)
        
        # Store max values for normalization/denormalization
        self.X_max = np.max(X)
        self.Y_max = np.max(Y)
        
        # Normalize
        X_normalized = X / self.X_max
        Y_normalized = Y / self.Y_max
        
        return X_normalized, Y_normalized
    
    def forward_pass(self, X):
        """
        Perform forward propagation through the network
        
        Args:
            X (numpy.ndarray): Input data
            
        Returns:
            tuple: Hidden layer output and final layer output
        """
        hidden_layer_input = np.dot(X, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        
        final_input = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
        final_output = self.sigmoid(final_input)
        
        return hidden_layer_output, final_output
    
    def backward_pass(self, X, Y, hidden_layer_output, final_output):
        """
        Perform backward propagation to update weights
        
        Args:
            X (numpy.ndarray): Input data
            Y (numpy.ndarray): Target data
            hidden_layer_output (numpy.ndarray): Output from hidden layer
            final_output (numpy.ndarray): Final output from network
        """
        # Calculate error
        error = Y - final_output
        
        # Calculate gradients
        d_output = error * self.sigmoid_derivative(final_output)
        error_hidden_layer = d_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)
        
        # Update weights and biases
        self.output_weights += hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.output_bias += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        
        self.hidden_weights += X.T.dot(d_hidden_layer) * self.learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            epochs (int): Number of training epochs
            verbose (bool): Whether to print progress
            
        Returns:
            list: History of loss values
        """
        if self.data is None:
            raise ValueError("Data not loaded and preprocessed. Call load_data() and preprocess_data() first.")
            
        X, Y = self.preprocess_data()
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            hidden_layer_output, final_output = self.forward_pass(X)
            
            # Calculate loss (MSE)
            loss = np.mean((Y - final_output) ** 2)
            loss_history.append(loss)
            
            # Backward pass
            self.backward_pass(X, Y, hidden_layer_output, final_output)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
        return loss_history
    
    def predict(self, height):
        """
        Make a prediction for a given height
        
        Args:
            height (float): Height in inches
            
        Returns:
            float: Predicted weight in pounds
        """
        if self.X_max is None or self.Y_max is None:
            raise ValueError("Model not trained. Train the model first.")
            
        # Normalize input
        height_normalized = height / self.X_max
        
        # Forward pass
        hidden_layer_output = self.sigmoid(np.dot(height_normalized, self.hidden_weights) + self.hidden_bias)
        final_input = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
        predicted_weight_normalized = self.sigmoid(final_input)
        
        # Denormalize output
        predicted_weight = predicted_weight_normalized * self.Y_max
        
        return predicted_weight.item(0)
    
    def save_weights(self, directory='models'):
        """
        Save the model weights and parameters
        
        Args:
            directory (str): Directory to save the weights
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        np.save(f'{directory}/hidden_weights.npy', self.hidden_weights)
        np.save(f'{directory}/hidden_bias.npy', self.hidden_bias)
        np.save(f'{directory}/output_weights.npy', self.output_weights)
        np.save(f'{directory}/output_bias.npy', self.output_bias)
        
        # Save normalization factors
        np.save(f'{directory}/height_max.npy', self.X_max)
        np.save(f'{directory}/weight_max.npy', self.Y_max)
        
        print(f"Model weights saved to '{directory}' directory")
    
    def load_weights(self, directory='models'):
        """
        Load the model weights and parameters
        
        Args:
            directory (str): Directory to load the weights from
        """
        import os
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' not found")
        
        self.hidden_weights = np.load(f'{directory}/hidden_weights.npy')
        self.hidden_bias = np.load(f'{directory}/hidden_bias.npy')
        self.output_weights = np.load(f'{directory}/output_weights.npy')
        self.output_bias = np.load(f'{directory}/output_bias.npy')
        
        # Load normalization factors
        self.X_max = np.load(f'{directory}/height_max.npy')
        self.Y_max = np.load(f'{directory}/weight_max.npy')
        
        print(f"Model weights loaded from '{directory}' directory")


# Main execution code (runs only when executing this file directly)
if __name__ == "__main__":
    # Create a neural network
    nn = NeuralNetwork()
    
    # Load data
    data = nn.load_data('height_weight.csv')
    
    # Train the model
    nn.train(epochs=1000)
    
    # Save the model weights
    nn.save_weights()
    
    # Test prediction
    sample_height = 70  # inches
    predicted_weight = nn.predict(sample_height)
    print(f"\nPredicted weight for height {sample_height} inches: {predicted_weight:.2f} pounds")
