import requests
import json
from datetime import datetime, timedelta
import numpy as np

BASE_URL = "http://localhost:5000"

def test_health_check():
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    print("Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_sample_data():
    print("Testing sample data generation...")
    response = requests.get(f"{BASE_URL}/generate_sample_data")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['data_points']} sample points")
        print(f"Prediction: {data['prediction']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_prediction():
    print("Testing prediction with custom data...")
    
    sample_data = []
    base_time = datetime.now()
    
    for i in range(40):
        timestamp = base_time - timedelta(minutes=15*i)
        sample_data.append({
            "timestamp": timestamp.isoformat(),
            "satellite_id": "G01",
            "orbit_error_m": 2.5 + 0.5 * np.sin(i * 0.1),
            "clock_error_ns": 15.0 + 2.0 * np.cos(i * 0.05),
            "radial_error_m": 1.8 + 0.3 * np.sin(i * 0.08),
            "ephemeris_age_hours": 2.0 + (i % 12) * 0.25
        })
    
    sample_data.reverse()
    
    payload = {"data": sample_data}
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        prediction = response.json()
        print("Prediction successful!")
        print(f"Predictions: {prediction['predictions']}")
        print(f"Horizon: {prediction['prediction_horizon']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_satellite_prediction():
    print("Testing satellite-specific prediction...")
    
    data = {
        "orbit_error_m": 2.5,
        "clock_error_ns": 15.0,
        "radial_error_m": 1.8,
        "ephemeris_age_hours": 2.0
    }
    
    response = requests.post(f"{BASE_URL}/predict/satellite/G01", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("Testing GNSS Prediction API")
    print("=" * 40)
    
    try:
        test_health_check()
        test_model_info()
        test_sample_data()
        test_prediction()
        test_satellite_prediction()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"Error during testing: {e}")