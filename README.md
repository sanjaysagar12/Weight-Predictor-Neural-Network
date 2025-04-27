# Neural Network Weight Predictor

A web application that uses a neural network model to predict weight based on height.

## Project Structure

```
/ann
  ├── ai.py               # Original neural network implementation
  ├── app.py              # Flask application
  ├── save_model.py       # Script to save model weights
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
   python save_model.py
   ```
4. Run the Flask application:
   ```
   python app.py
   ```
5. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
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
- Machine Learning: NumPy neural network implementation

## Extending the Project

- Add more input features to the neural network model
- Implement model versioning
- Add user authentication for prediction history
- Create a dashboard for visualizing prediction trends