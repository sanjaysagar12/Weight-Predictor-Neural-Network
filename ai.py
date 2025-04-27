import numpy as np
import pandas as pd

# 1. Load Data
data = pd.read_csv('height_weight.csv')  # replace with your filename
X = data['Height(Inches)'].values.reshape(-1, 1)
Y = data['Weight(Pounds)'].values.reshape(-1, 1)
print(X)
# 2. Normalize (important for neural nets!)
X = X / np.max(X)
Y = Y / np.max(Y)
print(X)
# 3. Activation Function
def sigmoid(x):
    # Clip values to avoid overflow
    x_safe = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_safe))

def sigmoid_derivative(x):
    return x * (1 - x)

# 4. Initialize Parameters
np.random.seed(42)
input_neurons = 1
hidden_neurons = 3
output_neurons = 1

# Random initial weights and biases
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# 5. Set Hyperparameters
learning_rate = 0.01
epochs = 1000

# 6. Training Loop
for epoch in range(epochs):
    # ---- Forward Pass ----
    hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    final_input = np.dot(hidden_layer_output, output_weights) + output_bias
    final_output = sigmoid(final_input)  # Use sigmoid activation for output layer

    # ---- Loss Calculation (Mean Squared Error) ----
    loss = np.mean((Y - final_output) ** 2)

    # ---- Backward Pass ----
    error = Y - final_output

    # derivative with respect to final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden_layer = d_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # ---- Update Weights and Biases ----
    output_weights += hidden_layer_output.T.dot(d_output) * learning_rate
    output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # ---- Print Loss ----
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 7. Test Prediction
sample_height = 60   # inches
sample_height_normalized = sample_height / np.max(data['Height(Inches)'])
hidden_layer_input = np.dot(sample_height_normalized, hidden_weights) + hidden_bias
hidden_layer_output = sigmoid(hidden_layer_input)
final_input = np.dot(hidden_layer_output, output_weights) + output_bias
predicted_weight = sigmoid(final_input) * np.max(data['Weight(Pounds)'])

# Extract scalar value properly from numpy array to avoid deprecation warning
predicted_weight_value = predicted_weight.item(0)
print(f"\nPredicted weight for height {sample_height} inches: {predicted_weight_value:.2f} pounds")
