from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models
ann_model = tf.keras.models.load_model("ANN_model.h5")
rnn_model = tf.keras.models.load_model("RNN_model.h5")
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_with_ann(input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = ann_model.predict(input_data_scaled)
    return float(prediction[0][0])

def predict_with_rnn(input_data):
    input_data_scaled = scaler.transform(input_data)
    input_data_reshaped = np.reshape(input_data_scaled, (input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
    prediction = rnn_model.predict(input_data_reshaped)
    return float(prediction[0][0])

def predict_with_rf(input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_data_scaled)
    return float(prediction[0])

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Get request parameters
        month = int(request.args.get("month"))
        day = int(request.args.get("day"))
        hour = int(request.args.get("hour"))
        temperature = float(request.args.get("temperature"))
        humidity = float(request.args.get("humidity"))
        ghi = float(request.args.get("ghi"))
        algorithm = request.args.get("algorithm")

        # Prepare input data for prediction
        input_data = np.array([[month, day, hour, temperature, humidity, ghi]])

        # Select the appropriate model based on the algorithm parameter
        if algorithm == "ANN":
            tilt_angle = predict_with_ann(input_data)
        elif algorithm == "RNN":
            tilt_angle = predict_with_rnn(input_data)
        elif algorithm == "RandomForest":
            tilt_angle = predict_with_rf(input_data)
        else:
            return jsonify({"error": "Invalid algorithm specified"}), 400

        return jsonify({"tilt_angle": tilt_angle}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
