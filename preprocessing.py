"""
GNSS Error Data Preprocessing Pipeline
=====================================

This module provides comprehensive preprocessing functionality for GNSS error prediction,
including normalization, categorical encoding, and time series sequence generation.

Features:
- Normalization of orbit_error_m and clock_error_ns
- Satellite ID encoding (embedding-based or one-hot)
- Time series sequence generation with configurable lookback windows
- Train/test split handling with proper data leakage prevention
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class GNSSErrorPreprocessor:
    """
    Comprehensive preprocessing pipeline for GNSS error prediction data.
    
    This class handles all preprocessing steps including normalization,
    categorical encoding, and sequence generation for time series modeling.
    """
    
    def __init__(self, 
                 normalization_method: str = 'standard',
                 encoding_method: str = 'embedding',
                 lookback_window: int = 12,
                 prediction_horizon: int = 1):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            normalization_method: 'standard' or 'minmax' normalization
            encoding_method: 'embedding' or 'onehot' for satellite_id encoding
            lookback_window: Number of time steps to look back (6-12 recommended)
            prediction_horizon: Number of time steps to predict ahead
        """
        self.normalization_method = normalization_method
        self.encoding_method = encoding_method
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Initialize scalers and encoders
        self.orbit_scaler = None
        self.clock_scaler = None
        self.label_encoder = None
        self.satellite_mapping = None
        
        # Store fit parameters
        self.is_fitted = False
        self.feature_names = None
        self.n_satellites = None
        
    def _initialize_scalers(self):
        """Initialize normalization scalers based on configuration."""
        if self.normalization_method == 'standard':
            self.orbit_scaler = StandardScaler()
            self.clock_scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.orbit_scaler = MinMaxScaler()
            self.clock_scaler = MinMaxScaler()
        else:
            raise ValueError("normalization_method must be 'standard' or 'minmax'")
    
    def _fit_encoders(self, df: pd.DataFrame):
        """Fit categorical encoders on satellite_id."""
        self.label_encoder = LabelEncoder()
        satellite_ids = df['satellite_id'].unique()
        self.label_encoder.fit(satellite_ids)
        self.n_satellites = len(satellite_ids)
        
        # Create satellite mapping for reference
        self.satellite_mapping = {
            sat_id: idx for idx, sat_id in enumerate(self.label_encoder.classes_)
        }
        
        print(f"Found {self.n_satellites} unique satellites: {sorted(satellite_ids)}")
    
    def _normalize_errors(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize orbit and clock errors.
        
        Args:
            df: DataFrame with orbit_error_m and clock_error_ns columns
            fit: Whether to fit the scalers (True for training, False for inference)
        
        Returns:
            DataFrame with normalized error columns
        """
        df_normalized = df.copy()
        
        if fit:
            self._initialize_scalers()
            
            # Fit and transform orbit errors
            orbit_errors = df[['orbit_error_m']].values
            orbit_errors_scaled = self.orbit_scaler.fit_transform(orbit_errors)
            
            # Fit and transform clock errors  
            clock_errors = df[['clock_error_ns']].values
            clock_errors_scaled = self.clock_scaler.fit_transform(clock_errors)
            
        else:
            if self.orbit_scaler is None or self.clock_scaler is None:
                raise ValueError("Scalers not fitted. Call fit() first.")
                
            # Transform only
            orbit_errors = df[['orbit_error_m']].values
            orbit_errors_scaled = self.orbit_scaler.transform(orbit_errors)
            
            clock_errors = df[['clock_error_ns']].values
            clock_errors_scaled = self.clock_scaler.transform(clock_errors)
        
        # Update dataframe with normalized values
        df_normalized['orbit_error_m_norm'] = orbit_errors_scaled.flatten()
        df_normalized['clock_error_ns_norm'] = clock_errors_scaled.flatten()
        
        return df_normalized
    
    def _encode_satellite_id(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode satellite_id using specified method.
        
        Args:
            df: DataFrame with satellite_id column
            fit: Whether to fit the encoder (True for training, False for inference)
        
        Returns:
            DataFrame with encoded satellite_id
        """
        df_encoded = df.copy()
        
        if fit:
            self._fit_encoders(df)
        
        if self.encoding_method == 'embedding':
            # For embedding, we just need integer labels
            df_encoded['satellite_id_encoded'] = self.label_encoder.transform(df['satellite_id'])
            
        elif self.encoding_method == 'onehot':
            # One-hot encoding
            satellite_encoded = self.label_encoder.transform(df['satellite_id'])
            
            # Create one-hot matrix
            onehot_matrix = np.eye(self.n_satellites)[satellite_encoded]
            
            # Add one-hot columns to dataframe
            for i, sat_id in enumerate(self.label_encoder.classes_):
                df_encoded[f'satellite_{sat_id}'] = onehot_matrix[:, i]
                
        else:
            raise ValueError("encoding_method must be 'embedding' or 'onehot'")
        
        return df_encoded
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series sequences for each satellite.
        
        Args:
            df: Preprocessed DataFrame with normalized and encoded features
        
        Returns:
            Tuple of (X_sequences, y_sequences) arrays
        """
        X_sequences = []
        y_sequences = []
        
        # Define feature columns based on encoding method
        if self.encoding_method == 'embedding':
            feature_cols = ['satellite_id_encoded', 'orbit_error_m_norm', 'clock_error_ns_norm', 'ephemeris_age_hours']
        else:  # onehot
            satellite_cols = [col for col in df.columns if col.startswith('satellite_')]
            feature_cols = satellite_cols + ['orbit_error_m_norm', 'clock_error_ns_norm', 'ephemeris_age_hours']
        
        # Target columns (what we want to predict)
        target_cols = ['orbit_error_m_norm', 'clock_error_ns_norm']
        
        # Process each satellite separately to maintain temporal continuity
        for satellite_id in df['satellite_id'].unique():
            sat_data = df[df['satellite_id'] == satellite_id].copy()
            sat_data = sat_data.sort_values('timestamp').reset_index(drop=True)
            
            # Extract features and targets
            features = sat_data[feature_cols].values
            targets = sat_data[target_cols].values
            
            # Create sequences
            for i in range(len(sat_data) - self.lookback_window - self.prediction_horizon + 1):
                # Input sequence (lookback_window time steps)
                X_seq = features[i:i + self.lookback_window]
                
                # Target sequence (prediction_horizon time steps ahead)
                y_seq = targets[i + self.lookback_window:i + self.lookback_window + self.prediction_horizon]
                
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
        
        # Convert to numpy arrays
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Store feature names for reference
        self.feature_names = feature_cols
        
        print(f"Created {len(X_sequences)} sequences")
        print(f"X shape: {X_sequences.shape} (samples, time_steps, features)")
        print(f"y shape: {y_sequences.shape} (samples, prediction_steps, targets)")
        
        return X_sequences, y_sequences
    
    def fit(self, df: pd.DataFrame) -> 'GNSSErrorPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame with columns: satellite_id, timestamp, 
                orbit_error_m, clock_error_ns, ephemeris_age_hours
        
        Returns:
            Self for method chaining
        """
        print("Fitting preprocessor...")
        print(f"Input data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Normalize errors
        df_normalized = self._normalize_errors(df, fit=True)
        
        # Encode satellite IDs
        df_encoded = self._encode_satellite_id(df_normalized, fit=True)
        
        self.is_fitted = True
        print("Preprocessor fitted successfully!")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data and create sequences.
        
        Args:
            df: DataFrame to transform
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        print("Transforming data...")
        
        # Normalize errors (using fitted scalers)
        df_normalized = self._normalize_errors(df, fit=False)
        
        # Encode satellite IDs (using fitted encoder)
        df_encoded = self._encode_satellite_id(df_normalized, fit=False)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(df_encoded)
        
        print("Data transformation completed!")
        
        return X_sequences, y_sequences
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        return self.fit(df).transform(df)
    
    def inverse_transform_targets(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized target values back to original scale.
        
        Args:
            y_normalized: Normalized target values (orbit_error_m_norm, clock_error_ns_norm)
        
        Returns:
            Target values in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Handle different input shapes
        original_shape = y_normalized.shape
        if len(original_shape) == 3:  # (samples, time_steps, features)
            y_reshaped = y_normalized.reshape(-1, y_normalized.shape[-1])
        else:
            y_reshaped = y_normalized
        
        # Inverse transform orbit errors (first column)
        orbit_errors = self.orbit_scaler.inverse_transform(y_reshaped[:, [0]])
        
        # Inverse transform clock errors (second column)
        clock_errors = self.clock_scaler.inverse_transform(y_reshaped[:, [1]])
        
        # Combine results
        y_original = np.concatenate([orbit_errors, clock_errors], axis=1)
        
        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            y_original = y_original.reshape(original_shape)
        
        return y_original
    
    def get_preprocessing_info(self) -> Dict:
        """
        Get information about the fitted preprocessor.
        
        Returns:
            Dictionary with preprocessing configuration and statistics
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        info = {
            "status": "fitted",
            "normalization_method": self.normalization_method,
            "encoding_method": self.encoding_method,
            "lookback_window": self.lookback_window,
            "prediction_horizon": self.prediction_horizon,
            "n_satellites": self.n_satellites,
            "satellite_mapping": self.satellite_mapping,
            "feature_names": self.feature_names,
        }
        
        # Add scaler statistics if available
        if hasattr(self.orbit_scaler, 'mean_'):
            info["orbit_scaler_stats"] = {
                "mean": float(self.orbit_scaler.mean_[0]),
                "scale": float(self.orbit_scaler.scale_[0])
            }
        
        if hasattr(self.clock_scaler, 'mean_'):
            info["clock_scaler_stats"] = {
                "mean": float(self.clock_scaler.mean_[0]),
                "scale": float(self.clock_scaler.scale_[0])
            }
        
        return info


def create_train_test_split(df: pd.DataFrame, 
                          test_size: float = 0.2, 
                          time_based: bool = True,
                          time_column: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split for time series data.
    
    Args:
        df: Input DataFrame
        test_size: Fraction of data to use for testing
        time_based: If True, split based on time (recommended for time series)
        time_column: Column to use for time-based splitting
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if time_based:
        # Sort by time
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        print(f"Time-based split:")
        print(f"  Train: {len(train_df)} samples ({train_df[time_column].min()} to {train_df[time_column].max()})")
        print(f"  Test:  {len(test_df)} samples ({test_df[time_column].min()} to {test_df[time_column].max()})")
        
    else:
        # Random split (not recommended for time series)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        print(f"Random split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
    
    return train_df, test_df


# Example usage and demonstration
if __name__ == "__main__":
    # Load the dataset
    print("Loading GNSS error dataset...")
    df = pd.read_csv('errors_day187_192.csv')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique satellites: {sorted(df['satellite_id'].unique())}")
    
    # Create train/test split (using day 193 as test set)
    train_df = df[df['timestamp'].dt.day != 13].copy()  # Days 187-192
    test_df = df[df['timestamp'].dt.day == 13].copy()   # Day 193
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Initialize preprocessor with different configurations
    configs = [
        {"lookback_window": 6, "encoding_method": "embedding"},
        {"lookback_window": 12, "encoding_method": "embedding"},
        {"lookback_window": 12, "encoding_method": "onehot"}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}: {config}")
        print(f"{'='*60}")
        
        # Initialize preprocessor
        preprocessor = GNSSErrorPreprocessor(**config)
        
        # Fit and transform training data
        X_train, y_train = preprocessor.fit_transform(train_df)
        
        # Transform test data
        X_test, y_test = preprocessor.transform(test_df)
        
        print(f"\nTraining sequences:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        
        print(f"\nTest sequences:")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        # Show preprocessing info
        info = preprocessor.get_preprocessing_info()
        print(f"\nPreprocessor info:")
        for key, value in info.items():
            if key not in ['satellite_mapping', 'feature_names']:
                print(f"  {key}: {value}")
        
        # Test inverse transformation
        y_sample = y_test[:5]  # Take first 5 samples
        y_original = preprocessor.inverse_transform_targets(y_sample)
        
        print(f"\nInverse transformation test:")
        print(f"  Normalized shape: {y_sample.shape}")
        print(f"  Original shape: {y_original.shape}")