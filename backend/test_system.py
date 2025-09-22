import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing model loading...")
    import tensorflow as tf
    
    model_path = "best_trained_lstm_model.keras"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    else:
        print(f"❌ Model file not found: {model_path}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTesting dependencies...")
try:
    import flask
    print("✅ Flask available")
except ImportError:
    print("❌ Flask not available")

try:
    import numpy as np
    print("✅ NumPy available")
except ImportError:
    print("❌ NumPy not available")

try:
    import pandas as pd
    print("✅ Pandas available")
except ImportError:
    print("❌ Pandas not available")

try:
    from sklearn.preprocessing import RobustScaler
    print("✅ Scikit-learn available")
except ImportError:
    print("❌ Scikit-learn not available")

print("\n🚀 System check complete!")