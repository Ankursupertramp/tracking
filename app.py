from flask import Flask, jsonify, request, send_from_directory
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging
import traceback

app = Flask(__name__, static_folder='.')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Load models at runtime to avoid startup issues
ann_model, rf_model, rnn_model, scaler = None, None, None, None

def load_models():
    global ann_model, rf_model, rnn_model, scaler
    try:
        app.logger.info("Loading models...")
        
        ann_model_path = os.getenv('ANN_MODEL_PATH', 'ANN_model.h5')
        ann_model = tf.keras.models.load_model(ann_model_path, custom_objects={'mse': mse})
        rf_model = joblib.load(os.getenv('RF_MODEL_PATH', 'random_forest_model.pkl'))
        rnn_model = tf.keras.models.load_model(os.getenv('RNN_MODEL_PATH', 'RNN_model.h5'), custom_objects={'mse': mse})
        scaler = joblib.load('scaler.pkl')
        
        app.logger.info("Models loaded successfully.")
    except Exception as e:
        app.logger.error(f"Error loading models: {e}")
        app.logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        if ann_model is None or rf_model is None or rnn_model is None or scaler is None:
            load_models()
        
        month = int(request.args.get('month'))
        day = int(request.args.get('day'))
        hour = int(request.args.get('hour'))
        temperature = float(request.args.get('temperature'))
        humidity = float(request.args.get('humidity'))
        ghi = float(request.args.get('ghi'))
        algorithm = request.args.get('algorithm', 'ANN')
        
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
        app.logger.error(f"Error in /predict endpoint: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)
