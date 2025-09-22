import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
from scipy.interpolate import interp1d


class ErrorCalculator:

    def __init__(self):

        self.C = 299792458.0
        
    def compute_orbit_error(self, broadcast_pos: Tuple[float, float, float], 
                          precise_pos: Tuple[float, float, float]) -> float:
        try:
            dx = broadcast_pos[0] - precise_pos[0]
            dy = broadcast_pos[1] - precise_pos[1]
            dz = broadcast_pos[2] - precise_pos[2]
            
            error_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            return error_3d
            
        except Exception as e:
            print(f"Error computing orbit error: {e}")
            return np.nan
    
    def compute_radial_along_cross_errors(self, broadcast_pos: Tuple[float, float, float], 
                                        precise_pos: Tuple[float, float, float],
                                        velocity: Optional[Tuple[float, float, float]] = None) -> Dict[str, float]:
        """
        Compute radial, along-track, and cross-track errors
        
        Args:
            broadcast_pos: Broadcast position (x, y, z) in meters
            precise_pos: Precise position (x, y, z) in meters
            velocity: Velocity vector (optional, for along/cross track)
            
        Returns:
            Dictionary with 'radial', 'along_track', 'cross_track' errors in meters
        """
        try:
            # Position difference vector
            dx = broadcast_pos[0] - precise_pos[0]
            dy = broadcast_pos[1] - precise_pos[1]
            dz = broadcast_pos[2] - precise_pos[2]
            error_vector = np.array([dx, dy, dz])
            
            # Unit vector from Earth center to precise position (radial direction)
            precise_vec = np.array(precise_pos)
            radial_unit = precise_vec / np.linalg.norm(precise_vec)
            
            # Radial error (positive = away from Earth)
            radial_error = np.dot(error_vector, radial_unit)
            
            # If velocity is provided, compute along-track and cross-track
            if velocity is not None:
                velocity_vec = np.array(velocity)
                velocity_unit = velocity_vec / np.linalg.norm(velocity_vec)
                
                # Along-track error
                along_track_error = np.dot(error_vector, velocity_unit)
                
                # Cross-track direction (perpendicular to both radial and along-track)
                cross_track_unit = np.cross(radial_unit, velocity_unit)
                cross_track_unit = cross_track_unit / np.linalg.norm(cross_track_unit)
                
                # Cross-track error
                cross_track_error = np.dot(error_vector, cross_track_unit)
                
                return {
                    'radial': radial_error,
                    'along_track': along_track_error,
                    'cross_track': cross_track_error
                }
            else:
                return {
                    'radial': radial_error,
                    'along_track': np.nan,
                    'cross_track': np.nan
                }
                
        except Exception as e:
            print(f"Error computing RAC errors: {e}")
            return {
                'radial': np.nan,
                'along_track': np.nan,
                'cross_track': np.nan
            }
    
    def compute_clock_error(self, broadcast_clock: float, precise_clock: float) -> float:
        """
        Compute clock error between broadcast and precise data
        
        Args:
            broadcast_clock: Clock bias from broadcast ephemeris (seconds)
            precise_clock: Clock bias from precise products (seconds)
            
        Returns:
            Clock error in nanoseconds
        """
        try:
            # Clock difference in seconds
            clock_diff = broadcast_clock - precise_clock
            
            # Convert to nanoseconds
            clock_error_ns = clock_diff * 1e9
            
            return clock_error_ns
            
        except Exception as e:
            print(f"Error computing clock error: {e}")
            return np.nan
    
    def interpolate_precise_data(self, precise_df: pd.DataFrame, 
                               target_timestamps: List[datetime],
                               satellite: str) -> pd.DataFrame:
        """
        Interpolate precise orbit and clock data to target timestamps
        
        Args:
            precise_df: DataFrame with precise orbit data
            target_timestamps: List of target timestamps for interpolation
            satellite: Satellite ID (e.g., 'G01')
            
        Returns:
            DataFrame with interpolated precise data
        """
        try:
            # Filter for specific satellite
            sat_data = precise_df[precise_df['satellite'] == satellite].copy()
            
            if sat_data.empty:
                return pd.DataFrame()
            
            # Sort by timestamp
            sat_data = sat_data.sort_values('timestamp')
            
            # Convert timestamps to numeric for interpolation
            time_numeric = np.array([(t - sat_data['timestamp'].iloc[0]).total_seconds() 
                                   for t in sat_data['timestamp']])
            target_numeric = np.array([(t - sat_data['timestamp'].iloc[0]).total_seconds() 
                                     for t in target_timestamps])
            
            # Filter target times to be within the data range
            min_time, max_time = time_numeric[0], time_numeric[-1]
            valid_mask = (target_numeric >= min_time) & (target_numeric <= max_time)
            valid_targets = target_numeric[valid_mask]
            valid_timestamps = [target_timestamps[i] for i in range(len(target_timestamps)) if valid_mask[i]]
            
            if len(valid_targets) == 0:
                return pd.DataFrame()
            
            # Interpolate position coordinates
            interp_results = {'timestamp': valid_timestamps, 'satellite': satellite}
            
            for coord in ['x', 'y', 'z']:
                if coord in sat_data.columns:
                    interp_func = interp1d(time_numeric, sat_data[coord], 
                                         kind='linear', bounds_error=False, fill_value=np.nan)
                    interp_results[f'{coord}_precise'] = interp_func(valid_targets)
            
            # Interpolate clock bias if available
            if 'clock_bias' in sat_data.columns:
                # Remove any NaN values for interpolation
                valid_clock = ~pd.isna(sat_data['clock_bias'])
                if valid_clock.sum() > 1:
                    clock_interp = interp1d(time_numeric[valid_clock], 
                                          sat_data['clock_bias'][valid_clock],
                                          kind='linear', bounds_error=False, 
                                          fill_value=np.nan)
                    interp_results['clock_precise'] = clock_interp(valid_targets)
                else:
                    interp_results['clock_precise'] = [np.nan] * len(valid_targets)
            
            return pd.DataFrame(interp_results)
            
        except Exception as e:
            print(f"Error interpolating precise data for {satellite}: {e}")
            return pd.DataFrame()
    
    def compute_errors_for_satellite(self, broadcast_df: pd.DataFrame,
                                   precise_orbit_df: pd.DataFrame,
                                   precise_clock_df: pd.DataFrame,
                                   satellite: str) -> pd.DataFrame:
        """
        Compute errors for a specific satellite across all available timestamps
        
        Args:
            broadcast_df: DataFrame with broadcast positions
            precise_orbit_df: DataFrame with precise orbit data
            precise_clock_df: DataFrame with precise clock data
            satellite: Satellite ID (e.g., 'G01')
            
        Returns:
            DataFrame with computed errors
        """
        try:
            # Filter broadcast data for this satellite
            sat_broadcast = broadcast_df[broadcast_df['satellite'] == satellite].copy()
            
            if sat_broadcast.empty:
                return pd.DataFrame()
            
            # Get timestamps from broadcast data
            timestamps = sat_broadcast['timestamp'].tolist()
            
            # Interpolate precise orbit data
            precise_orbit_interp = self.interpolate_precise_data(
                precise_orbit_df, timestamps, satellite
            )
            
            # Interpolate precise clock data
            precise_clock_interp = self.interpolate_precise_data(
                precise_clock_df, timestamps, satellite
            )
            
            # Merge all data
            merged_data = sat_broadcast.merge(precise_orbit_interp, 
                                            on=['timestamp', 'satellite'], 
                                            how='inner')
            
            if not precise_clock_interp.empty:
                merged_data = merged_data.merge(precise_clock_interp[['timestamp', 'satellite', 'clock_precise']], 
                                              on=['timestamp', 'satellite'], 
                                              how='left')
            
            if merged_data.empty:
                return pd.DataFrame()
            
            # Compute errors
            error_results = []
            
            for _, row in merged_data.iterrows():
                # Check if we have valid precise orbit data
                if any(pd.isna(row[coord]) for coord in ['x_precise', 'y_precise', 'z_precise']):
                    continue
                
                # Orbit positions
                broadcast_pos = (row['x_broadcast'], row['y_broadcast'], row['z_broadcast'])
                precise_pos = (row['x_precise'], row['y_precise'], row['z_precise'])
                
                # Compute 3D orbit error
                orbit_error = self.compute_orbit_error(broadcast_pos, precise_pos)
                
                # Compute radial/along/cross errors
                rac_errors = self.compute_radial_along_cross_errors(broadcast_pos, precise_pos)
                
                # Compute clock error if available
                clock_error_ns = np.nan
                if 'clock_broadcast' in row and 'clock_precise' in row:
                    if not pd.isna(row['clock_broadcast']) and not pd.isna(row['clock_precise']):
                        clock_error_ns = self.compute_clock_error(
                            row['clock_broadcast'], row['clock_precise']
                        )
                
                error_results.append({
                    'timestamp': row['timestamp'],
                    'satellite': satellite,
                    'orbit_error_m': orbit_error,
                    'radial_error_m': rac_errors['radial'],
                    'along_track_error_m': rac_errors['along_track'],
                    'cross_track_error_m': rac_errors['cross_track'],
                    'clock_error_ns': clock_error_ns,
                    'ephemeris_age_hours': row.get('ephemeris_age', np.nan)
                })
            
            return pd.DataFrame(error_results)
            
        except Exception as e:
            print(f"Error computing errors for {satellite}: {e}")
            return pd.DataFrame()
    
    def compute_all_errors(self, broadcast_df: pd.DataFrame,
                         precise_orbit_df: pd.DataFrame,
                         precise_clock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute errors for all satellites
        
        Args:
            broadcast_df: DataFrame with broadcast positions
            precise_orbit_df: DataFrame with precise orbit data
            precise_clock_df: DataFrame with precise clock data
            
        Returns:
            DataFrame with all computed errors
        """
        all_errors = []
        
        # Get unique satellites from broadcast data
        satellites = broadcast_df['satellite'].unique()
        gps_satellites = [sat for sat in satellites if sat.startswith('G')]
        
        print(f"Computing errors for {len(gps_satellites)} GPS satellites")
        
        for i, satellite in enumerate(gps_satellites):
            print(f"  Processing {satellite} ({i+1}/{len(gps_satellites)})")
            
            sat_errors = self.compute_errors_for_satellite(
                broadcast_df, precise_orbit_df, precise_clock_df, satellite
            )
            
            if not sat_errors.empty:
                all_errors.append(sat_errors)
                print(f"    Computed {len(sat_errors)} error records")
            else:
                print(f"    No valid error records for {satellite}")
        
        if all_errors:
            combined_errors = pd.concat(all_errors, ignore_index=True)
            combined_errors = combined_errors.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            print(f"Total error records: {len(combined_errors)}")
            return combined_errors
        else:
            return pd.DataFrame()
    
    def generate_error_statistics(self, error_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the error dataset
        
        Args:
            error_df: DataFrame with computed errors
            
        Returns:
            Dictionary with error statistics
        """
        if error_df.empty:
            return {}
        
        stats = {}
        
        # Overall statistics
        stats['total_records'] = len(error_df)
        stats['unique_satellites'] = error_df['satellite'].nunique()
        stats['time_span'] = {
            'start': error_df['timestamp'].min(),
            'end': error_df['timestamp'].max(),
            'duration_hours': (error_df['timestamp'].max() - error_df['timestamp'].min()).total_seconds() / 3600
        }
        
        # Orbit error statistics
        orbit_errors = error_df['orbit_error_m'].dropna()
        if not orbit_errors.empty:
            stats['orbit_error'] = {
                'mean': float(orbit_errors.mean()),
                'std': float(orbit_errors.std()),
                'min': float(orbit_errors.min()),
                'max': float(orbit_errors.max()),
                'median': float(orbit_errors.median()),
                'percentile_95': float(orbit_errors.quantile(0.95))
            }
        
        # Clock error statistics
        clock_errors = error_df['clock_error_ns'].dropna()
        if not clock_errors.empty:
            stats['clock_error'] = {
                'mean': float(clock_errors.mean()),
                'std': float(clock_errors.std()),
                'min': float(clock_errors.min()),
                'max': float(clock_errors.max()),
                'median': float(clock_errors.median()),
                'percentile_95': float(clock_errors.quantile(0.95))
            }
        
        # Per-satellite statistics
        stats['per_satellite'] = {}
        for satellite in error_df['satellite'].unique():
            sat_data = error_df[error_df['satellite'] == satellite]
            sat_orbit = sat_data['orbit_error_m'].dropna()
            sat_clock = sat_data['clock_error_ns'].dropna()
            
            stats['per_satellite'][satellite] = {
                'records': len(sat_data),
                'orbit_error_mean': float(sat_orbit.mean()) if not sat_orbit.empty else np.nan,
                'clock_error_mean': float(sat_clock.mean()) if not sat_clock.empty else np.nan
            }
        
        return stats


def test_error_calculator():
    """Test function to verify error calculation functionality"""
    calculator = ErrorCalculator()
    
    print("Testing error calculation functions...")
    
    # Test orbit error calculation
    broadcast_pos = (26560000.0, 1000000.0, 0.0)  # Example position
    precise_pos = (26560100.0, 1000200.0, 50.0)   # Slightly different position
    
    orbit_error = calculator.compute_orbit_error(broadcast_pos, precise_pos)
    print(f"Test orbit error: {orbit_error:.2f} meters")
    
    # Test radial/along/cross errors
    rac_errors = calculator.compute_radial_along_cross_errors(broadcast_pos, precise_pos)
    print(f"Radial error: {rac_errors['radial']:.2f} meters")
    
    # Test clock error calculation
    broadcast_clock = 1.23e-6  # 1.23 microseconds
    precise_clock = 1.25e-6    # 1.25 microseconds
    
    clock_error = calculator.compute_clock_error(broadcast_clock, precise_clock)
    print(f"Test clock error: {clock_error:.2f} nanoseconds")
    
    # Test interpolation with sample data
    sample_data = pd.DataFrame({
        'timestamp': [datetime(2025, 7, 6, 12, 0, 0) + timedelta(minutes=i*15) for i in range(5)],
        'satellite': ['G01'] * 5,
        'x': [26560000 + i*1000 for i in range(5)],
        'y': [1000000 + i*500 for i in range(5)],
        'z': [0 + i*100 for i in range(5)]
    })
    
    target_times = [datetime(2025, 7, 6, 12, 0, 0) + timedelta(minutes=i*7.5) for i in range(9)]
    
    interpolated = calculator.interpolate_precise_data(sample_data, target_times, 'G01')
    print(f"Interpolation test: {len(interpolated)} records interpolated from {len(sample_data)} input records")
    
    if not interpolated.empty:
        print("✓ Error calculation tests passed")
    else:
        print("⚠ Some error calculation tests failed")


if __name__ == "__main__":
    test_error_calculator()