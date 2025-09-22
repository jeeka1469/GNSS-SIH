import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import json
from datetime import datetime, timedelta
import pickle
import urllib.request
from pathlib import Path

app = Flask(__name__)
CORS(app)

class GNSSPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = [
            'orbit_error_m', 'clock_error_ns', 'radial_error_m', 'ephemeris_age_hours',
            'day_sin', 'day_cos',
            'orbit_error_m_ma', 'clock_error_ns_ma', 'radial_error_m_ma',
            'orbit_error_m_std', 'clock_error_ns_std', 'radial_error_m_std',
            'orbit_error_m_diff', 'clock_error_ns_diff', 'radial_error_m_diff'
        ]
        self.load_model()
    
    def load_model(self):
        try:
            # Try local path first (for Railway deployment)
            model_path = os.path.join(os.path.dirname(__file__), 'best_trained_lstm_model.keras')
            
            # If model doesn't exist locally, try to download it
            if not os.path.exists(model_path):
                print("Model not found locally, checking for download URL...")
                # You can set this as an environment variable in Railway
                model_url = os.environ.get('MODEL_DOWNLOAD_URL')
                if model_url:
                    print(f"Downloading model from: {model_url}")
                    try:
                        urllib.request.urlretrieve(model_url, model_path)
                        print("Model downloaded successfully")
                    except Exception as download_error:
                        print(f"Failed to download model: {download_error}")
                        self.model = None
                        return
                else:
                    # Fallback to parent directory
                    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_trained_lstm_model.keras')
            
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                # Load with custom objects to handle compatibility issues
                self.model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=None,
                    compile=False
                )
                # Recompile the model to ensure compatibility
                self.model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                print("Model loaded successfully")
            else:
                print("Model file not found")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try alternative loading method
            try:
                print("Trying alternative loading method...")
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                print("Model loaded with alternative method")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                self.model = None
    
    def create_enhanced_features(self, data):
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
        else:
            df['hour'] = np.arange(len(df)) % 24
        
        df['day_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['day_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        window = 12
        for col in ['orbit_error_m', 'clock_error_ns', 'radial_error_m']:
            if col in df.columns:
                df[f'{col}_ma'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'{col}_diff'] = df[col].diff().fillna(0)
        
        return df[self.feature_cols].fillna(0)
    
    def predict(self, data):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            enhanced_data = self.create_enhanced_features(data)
            
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(enhanced_data.values.astype('float32'))
            
            seq_len = 36
            if len(scaled_data) < seq_len:
                return {"error": f"Need at least {seq_len} data points"}
            
            X = scaled_data[-seq_len:].reshape(1, seq_len, -1)
            
            prediction = self.model.predict(X, verbose=0)
            
            target_features = 4
            pred_padded = np.zeros((prediction.shape[0] * prediction.shape[1], len(self.feature_cols)))
            pred_padded[:, :target_features] = prediction.reshape(-1, target_features)
            
            pred_original = scaler.inverse_transform(pred_padded)[:, :target_features]
            
            result = {
                "success": True,
                "predictions": {
                    "orbit_error_m": float(pred_original[0, 0]),
                    "clock_error_ns": float(pred_original[0, 1]),
                    "radial_error_m": float(pred_original[0, 2]),
                    "ephemeris_age_hours": float(pred_original[0, 3])
                },
                "prediction_horizon": "1 hour ahead",
                "model_confidence": "High"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

predictor = GNSSPredictor()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "GNSS Error Prediction API",
        "model_loaded": predictor.model is not None,
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({"error": "Missing 'data' field in request"}), 400
        
        result = predictor.predict(data['data'])
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/predict/satellite/<satellite_id>', methods=['POST'])
def predict_satellite(satellite_id):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        data['satellite_id'] = satellite_id
        result = predictor.predict([data])
        
        if "error" in result:
            return jsonify(result), 400
        
        result['satellite_id'] = satellite_id
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/generate_sample_data', methods=['GET'])
def generate_sample_data():
    try:
        timestamps = []
        base_time = datetime.now()
        
        sample_data = []
        for i in range(50):
            timestamp = base_time - timedelta(minutes=15*i)
            
            orbit_error = 2.5 + 0.5 * np.sin(i * 0.1) + np.random.normal(0, 0.1)
            clock_error = 15.0 + 2.0 * np.cos(i * 0.05) + np.random.normal(0, 0.5)
            radial_error = 1.8 + 0.3 * np.sin(i * 0.08) + np.random.normal(0, 0.05)
            ephemeris_age = 2.0 + (i % 12) * 0.25
            
            sample_data.append({
                "timestamp": timestamp.isoformat(),
                "satellite_id": "G01",
                "orbit_error_m": round(orbit_error, 3),
                "clock_error_ns": round(clock_error, 3),
                "radial_error_m": round(radial_error, 3),
                "ephemeris_age_hours": round(ephemeris_age, 2)
            })
        
        sample_data.reverse()
        
        prediction_result = predictor.predict(sample_data)
        
        return jsonify({
            "sample_data": sample_data,
            "prediction": prediction_result,
            "data_points": len(sample_data),
            "satellite": "G01"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate sample data: {str(e)}"}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": "LSTM Neural Network",
        "architecture": "3-layer LSTM (256->128->64) + Dense layers",
        "features": 15,
        "sequence_length": 36,
        "prediction_horizon": 4,
        "target_variables": ["orbit_error_m", "clock_error_ns", "radial_error_m", "ephemeris_age_hours"],
        "training_data": "GNSS satellite error data (7 days)",
        "performance": "High accuracy time series prediction"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
