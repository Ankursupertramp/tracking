from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load
import pickle
import os

app = Flask(__name__)

# Load the models
ann_model = load_model('ANN_model.h5')
rnn_model = load_model('RNN_model.h5')
rf_model = load('random_forest_model.pkl')

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = [float(x) for x in request.form.values()]
        data = np.array(data).reshape(1, -1)

        # Scale the input data
        scaled_data = scaler.transform(data)

        # Select the model based on user's choice
        model_type = request.form.get('model_type')
        
        if model_type == 'ANN':
            prediction = ann_model.predict(scaled_data)
        elif model_type == 'RNN':
            # RNN expects a 3D input (samples, timesteps, features)
            rnn_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
            prediction = rnn_model.predict(rnn_data)
        elif model_type == 'RandomForest':
            prediction = rf_model.predict(scaled_data)
        else:
            return jsonify({'error': 'Invalid model selected'})

        output = prediction[0][0] if model_type in ['ANN', 'RNN'] else prediction[0]
        
        return render_template('index.html', prediction_text=f'Predicted Tilt Angle: {output:.2f}')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Specify the port Render expects
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
