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

# Load models at startup
app.logger.info("Loading models...")
try:
    app.logger.info(f"TensorFlow version: {tf.__version__}")
    app.logger.info(f"Current working directory: {os.getcwd()}")
    app.logger.info(f"Files in current directory: {os.listdir('.')}")

    ann_model = tf.keras.models.load_model('ANN_model.h5', custom_objects={'mse': mse})
    app.logger.info("ANN model loaded successfully")
    
    rf_model = joblib.load('random_forest_model.pkl')
    app.logger.info("Random Forest model loaded successfully")
    
    scaler = joblib.load('scaler.pkl')
    app.logger.info("Scaler loaded successfully")
    
    rnn_model = tf.keras.models.load_model('RNN_model.h5', custom_objects={'mse': mse})
    app.logger.info("RNN model loaded successfully")
    
    app.logger.info("All models loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading models: {e}")
    app.logger.error(traceback.format_exc())
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
            try:
                if model == rnn_model:
                    app.logger.debug("Using RNN model for prediction.")
                    input_sequence = np.repeat(input_scaled, 24, axis=0)
                    input_sequence = np.expand_dims(input_sequence, axis=0)
                    app.logger.debug(f"Input sequence shape for RNN: {input_sequence.shape}")
                    predicted_tilt_angle = model.predict(input_sequence, batch_size=1)[0][0]
                else:  # ANN model
                    app.logger.debug("Using ANN model for prediction.")
                    app.logger.debug(f"Input shape for ANN: {input_scaled.shape}")
                    app.logger.debug("Starting ANN prediction...")
                    predicted_tilt_angle = model.predict(input_scaled, batch_size=1)[0][0]
                    app.logger.debug("ANN prediction completed.")
                
                app.logger.debug(f"Raw predicted tilt angle: {predicted_tilt_angle}")
                
                if np.isnan(predicted_tilt_angle) or np.isinf(predicted_tilt_angle):
                    raise ValueError("Invalid prediction: NaN or Inf")
            except Exception as model_error:
                app.logger.error(f"Error in TensorFlow model prediction: {model_error}")
                app.logger.error(traceback.format_exc())
                raise
        else:  # Random Forest model
            app.logger.debug("Model is a Random Forest Model.")
            predicted_tilt_angle = model.predict(input_scaled)[0]
        
        app.logger.debug(f"Predicted tilt angle: {predicted_tilt_angle}")
        
        # Adjust angle based on hour
        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle
        
        return float(predicted_tilt_angle)
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        app.logger.error(traceback.format_exc())
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
        app.logger.error(f"Error in /predict endpoint: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)
