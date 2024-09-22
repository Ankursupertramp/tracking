from flask import Flask, jsonify, request, send_from_directory
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='.')

def mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Load models at startup
print("Loading models...")
ann_model = tf.keras.models.load_model('ANN_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError})
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
rnn_model = tf.keras.models.load_model('RNN_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError})
print("Models loaded successfully.")

def predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi):
    try:
        input_data = pd.DataFrame({
            'Month': [month],
            'Day': [day],
            'Hour': [hour],
            'Temperature': [temperature],
            'Relative Humidity': [humidity],
            'GHI': [ghi]
        })

        input_scaled = scaler.transform(input_data)

        if isinstance(model, tf.keras.Model):
            if model == rnn_model:
                input_sequence = np.repeat(input_scaled, 24, axis=0)
                input_sequence = np.expand_dims(input_sequence, axis=0)
                predicted_tilt_angle = model.predict(input_sequence)[0][0]
            else:  # ANN model
                predicted_tilt_angle = model.predict(input_scaled)[0][0]
        else:  # Random Forest model
            predicted_tilt_angle = model.predict(input_scaled)[0]

        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle

        return float(predicted_tilt_angle)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        month = request.args.get('month', type=int)
        day = request.args.get('day', type=int)
        hour = request.args.get('hour', type=int)
        temperature = request.args.get('temperature', type=float)
        humidity = request.args.get('humidity', type=float)
        ghi = request.args.get('ghi', type=float)
        algorithm = request.args.get('algorithm', type=str, default='ANN')

        if None in (month, day, hour, temperature, humidity, ghi):
            return jsonify({'error': 'Missing or invalid query parameters'}), 400

        if algorithm == 'ANN':
            model = ann_model
        elif algorithm == 'RandomForest':
            model = rf_model
        elif algorithm == 'RNN':
            model = rnn_model
        else:
            return jsonify({'error': 'Invalid algorithm selection'}), 400

        tilt_angle = predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi)

        if tilt_angle is None:
            return jsonify({'error': 'Error in prediction'}), 500

        return jsonify({'angle': tilt_angle})

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)
