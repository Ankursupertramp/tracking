from flask import Flask, jsonify, request, send_from_directory
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging

app = Flask(__name__, static_folder='.')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Load models at startup
app.logger.info("Loading models...")
try:
    ann_model = tf.keras.models.load_model('ANN_model.h5', custom_objects={'mse': mse})
    rf_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    rnn_model = tf.keras.models.load_model('RNN_model.h5', custom_objects={'mse': mse})
    app.logger.info("Models loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading models: {e}")
    raise

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
        
        app.logger.debug(f"Input data: {input_data}")
        app.logger.debug(f"Scaled input: {input_scaled}")
        
        predicted_tilt_angle = None
        
        if isinstance(model, tf.keras.Model):
            app.logger.debug("Model is a TensorFlow Keras Model.")
            if model == rnn_model:
                app.logger.debug("Using RNN model for prediction.")
                input_sequence = np.repeat(input_scaled, 24, axis=0)
                input_sequence = np.expand_dims(input_sequence, axis=0)
                app.logger.debug(f"Input sequence shape for RNN: {input_sequence.shape}")
                predicted_tilt_angle = model.predict(input_sequence)[0][0]
            else:  # ANN model
                app.logger.debug("Using ANN model for prediction.")
                app.logger.debug(f"Input shape for ANN: {input_scaled.shape}")
                predicted_tilt_angle = model.predict(input_scaled)[0][0]
                app.logger.debug(f"Predicted tilt angle (ANN): {predicted_tilt_angle}")
        else:  # Random Forest model
            app.logger.debug("Model is a Random Forest Model.")
            predicted_tilt_angle = model.predict(input_scaled)[0]
            app.logger.debug(f"Predicted tilt angle (RF): {predicted_tilt_angle}")
        
        # Adjust angle based on hour
        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle
        
        return float(predicted_tilt_angle)
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}", exc_info=True)
        return None


@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        month = int(request.args.get('month'))
        day = int(request.args.get('day'))
        hour = int(request.args.get('hour'))
        temperature = float(request.args.get('temperature'))
        humidity = float(request.args.get('humidity'))
        ghi = float(request.args.get('ghi'))
        algorithm = request.args.get('algorithm', 'ANN')
        
        app.logger.info(f"Received request: month={month}, day={day}, hour={hour}, temperature={temperature}, humidity={humidity}, ghi={ghi}, algorithm={algorithm}")
        
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
        
        app.logger.info(f"Predicted tilt angle: {tilt_angle}")
        return jsonify({'angle': tilt_angle})
    except Exception as e:
        app.logger.error(f"Error in /predict endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)
