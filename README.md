# Neural Network Weight Predictor

A web application that uses a neural network model to predict weight based on height.

## Project Structure

```
/ann
  ├── ai.py               # Neural network class implementation
  ├── app.py              # Flask application
  ├── save_model.py       # Script to save model weights with CLI arguments
  ├── requirements.txt    # Python dependencies
  ├── models/             # Directory for saved model weights (created by save_model.py)
  └── templates/
      └── index.html      # Frontend HTML with Tailwind CSS and jQuery
```

## Setup Instructions

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Save the trained model:
   ```
   python save_model.py train
   ```
4. Run the Flask application:
   ```
   python app.py
   ```
5. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Command-Line Arguments

The save_model.py script now supports command-line arguments for easier model training and testing:

### Commands
- `train`: Train and save a new model
- `test`: Load and test an existing model
- `both`: Train a new model and then test it

### Options
- `--epochs`: Number of training epochs (default: 1000)
- `--hidden`: Number of hidden neurons (default: 3)
- `--lr`: Learning rate (default: 0.01)
- `--data`: Path to data file (default: height_weight.csv)

### Examples
```bash
# Train with default parameters
python save_model.py train

# Train with custom parameters
python save_model.py train --epochs 2000 --hidden 5 --lr 0.005

# Test an existing model
python save_model.py test

# Train and then test in one command
python save_model.py both --epochs 1500 --hidden 4

# Use a different data file
python save_model.py train --data custom_data.csv
```

## API Endpoints

- `GET /` - Returns the web interface
- `POST /api/predict` - Makes a weight prediction based on height input
  - Request body: `{ "height": 70 }`
  - Response: `{ "height": 70, "predicted_weight": 170.92, "success": true }`
- `GET /api/health` - Health check endpoint
- `POST /api/reload-model` - Reloads the model weights without restarting the server

## Technologies Used

- Backend: Flask (Python)
- Frontend: HTML, Tailwind CSS, jQuery
- Machine Learning: Object-oriented neural network implementation
- User Interface: Height input in CM, weight output in KG (with conversions)

## Extending the Project

- Add more input features to the neural network model
- Implement model versioning
- Add user authentication for prediction history
- Create a dashboard for visualizing prediction trends