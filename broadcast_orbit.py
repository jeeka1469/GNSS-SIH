import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math


class BroadcastOrbitComputer:
    """Computes satellite positions from broadcast ephemeris parameters"""
    
    def __init__(self):
        """Initialize with GPS constants"""
        # GPS Constants (from ICD-GPS-200)
        self.GM = 3.986005e14  # Earth's gravitational parameter (m³/s²)
        self.OMEGA_E = 7.2921151467e-5  # Earth rotation rate (rad/s)
        self.C = 299792458.0  # Speed of light (m/s)
        self.F = -4.442807633e-10  # Clock relativistic correction coefficient (s/m^0.5)
        
        # Reference time constants
        self.GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)  # GPS epoch start
        self.SECONDS_PER_WEEK = 604800.0  # Seconds in a GPS week
        self.SECONDS_PER_DAY = 86400.0
        
    def gps_time_to_seconds(self, timestamp: datetime) -> float:
        """Convert datetime to GPS seconds from GPS epoch"""
        delta = timestamp - self.GPS_EPOCH
        return delta.total_seconds()
    
    def compute_satellite_position(self, eph: Dict, timestamp: datetime) -> Optional[Tuple[float, float, float]]:
        """
        Compute satellite position at given timestamp using broadcast ephemeris
        
        Args:
            eph: Dictionary containing ephemeris parameters
            timestamp: Target time for computation
            
        Returns:
            Tuple of (x, y, z) coordinates in meters, or None if computation fails
        """
        try:
            # Check if we have required parameters
            required_params = ['a', 'e', 'i0', 'omega0', 'omega', 'm0', 'delta_n', 
                             'omega_dot', 'idot', 'cuc', 'cus', 'crc', 'crs', 'cic', 'cis', 'toe']
            
            # Use fallback values if some parameters are missing
            for param in required_params:
                if param not in eph or pd.isna(eph[param]):
                    if param in ['cuc', 'cus', 'crc', 'crs', 'cic', 'cis', 'delta_n', 'omega_dot', 'idot']:
                        eph[param] = 0.0  # Correction terms can default to 0
                    else:
                        print(f"Missing critical parameter: {param}")
                        return None
            
            # Time from ephemeris reference epoch
            t = self.gps_time_to_seconds(timestamp)
            toe = eph['toe']
            tk = t - toe
            
            # Handle week crossover
            if tk > 302400:
                tk -= self.SECONDS_PER_WEEK
            elif tk < -302400:
                tk += self.SECONDS_PER_WEEK
            
            # Semi-major axis
            a = eph['a']
            if a <= 0:
                return None
                
            # Computed mean motion
            n0 = math.sqrt(self.GM / (a**3))
            
            # Corrected mean motion
            n = n0 + eph['delta_n']
            
            # Mean anomaly
            Mk = eph['m0'] + n * tk
            
            # Solve Kepler's equation for eccentric anomaly
            Ek = self.solve_kepler_equation(Mk, eph['e'])
            
            # True anomaly
            vk = math.atan2(math.sqrt(1 - eph['e']**2) * math.sin(Ek), 
                           math.cos(Ek) - eph['e'])
            
            # Argument of latitude
            Phik = vk + eph['omega']
            
            # Second harmonic perturbations
            duk = eph['cus'] * math.sin(2 * Phik) + eph['cuc'] * math.cos(2 * Phik)
            drk = eph['crs'] * math.sin(2 * Phik) + eph['crc'] * math.cos(2 * Phik)
            dik = eph['cis'] * math.sin(2 * Phik) + eph['cic'] * math.cos(2 * Phik)
            
            # Corrected argument of latitude, radius, and inclination
            uk = Phik + duk
            rk = a * (1 - eph['e'] * math.cos(Ek)) + drk
            ik = eph['i0'] + dik + eph['idot'] * tk
            
            # Positions in orbital plane
            xk_prime = rk * math.cos(uk)
            yk_prime = rk * math.sin(uk)
            
            # Corrected longitude of ascending node
            Omegak = eph['omega0'] + (eph['omega_dot'] - self.OMEGA_E) * tk - self.OMEGA_E * toe
            
            # Earth-fixed coordinates
            xk = xk_prime * math.cos(Omegak) - yk_prime * math.cos(ik) * math.sin(Omegak)
            yk = xk_prime * math.sin(Omegak) + yk_prime * math.cos(ik) * math.cos(Omegak)
            zk = yk_prime * math.sin(ik)
            
            return (xk, yk, zk)
            
        except Exception as e:
            print(f"Error computing satellite position: {e}")
            return None
    
    def solve_kepler_equation(self, M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
        """
        Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E
        
        Args:
            M: Mean anomaly (radians)
            e: Eccentricity
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Eccentric anomaly E (radians)
        """
        # Initial guess
        E = M
        
        for i in range(max_iter):
            f = E - e * math.sin(E) - M
            if abs(f) < tol:
                break
            fp = 1 - e * math.cos(E)
            E = E - f / fp
            
        return E
    
    def compute_satellite_clock_correction(self, eph: Dict, timestamp: datetime) -> float:
        """
        Compute satellite clock correction from broadcast parameters
        
        Args:
            eph: Dictionary containing ephemeris parameters
            timestamp: Target time for computation
            
        Returns:
            Clock correction in seconds
        """
        try:
            # Time from clock reference epoch
            t = self.gps_time_to_seconds(timestamp)
            toc = eph.get('toc', eph.get('toe', t))  # Use toe if toc not available
            dt = t - toc
            
            # Handle week crossover
            if dt > 302400:
                dt -= self.SECONDS_PER_WEEK
            elif dt < -302400:
                dt += self.SECONDS_PER_WEEK
            
            # Clock polynomial correction
            af0 = eph.get('af0', 0.0)
            af1 = eph.get('af1', 0.0)
            af2 = eph.get('af2', 0.0)
            
            dtclk = af0 + af1 * dt + af2 * dt**2
            
            # Relativistic correction (if we have orbital parameters)
            if all(param in eph for param in ['a', 'e']):
                # Compute mean motion for relativistic correction
                n0 = math.sqrt(self.GM / (eph['a']**3))
                Mk = eph.get('m0', 0.0) + n0 * dt
                Ek = self.solve_kepler_equation(Mk, eph['e'])
                
                # Relativistic correction
                dtr = self.F * eph['e'] * math.sqrt(eph['a']) * math.sin(Ek)
                dtclk += dtr
            
            return dtclk
            
        except Exception as e:
            print(f"Error computing clock correction: {e}")
            return 0.0
    
    def compute_positions_for_timestamps(self, nav_df: pd.DataFrame, 
                                       timestamps: List[datetime], 
                                       satellite: str) -> pd.DataFrame:
        """
        Compute broadcast positions for a satellite at multiple timestamps
        
        Args:
            nav_df: Navigation data DataFrame
            timestamps: List of target timestamps
            satellite: Satellite ID (e.g., 'G01')
            
        Returns:
            DataFrame with computed positions and clock corrections
        """
        results = []
        
        for timestamp in timestamps:
            # Find the appropriate ephemeris for this timestamp
            sat_nav = nav_df[nav_df['satellite'] == satellite].copy()
            
            if sat_nav.empty:
                continue
                
            # Find closest ephemeris (within reasonable time)
            sat_nav['time_diff'] = abs(sat_nav['timestamp'] - timestamp)
            valid_eph = sat_nav[sat_nav['time_diff'] <= timedelta(hours=2)]
            
            if valid_eph.empty:
                continue
                
            # Use the closest ephemeris
            closest_idx = valid_eph['time_diff'].idxmin()
            eph = valid_eph.loc[closest_idx].to_dict()
            
            # Compute position
            position = self.compute_satellite_position(eph, timestamp)
            if position is None:
                continue
                
            # Compute clock correction
            clock_corr = self.compute_satellite_clock_correction(eph, timestamp)
            
            results.append({
                'timestamp': timestamp,
                'satellite': satellite,
                'x_broadcast': position[0],
                'y_broadcast': position[1],
                'z_broadcast': position[2],
                'clock_broadcast': clock_corr,
                'ephemeris_age': valid_eph.loc[closest_idx, 'time_diff'].total_seconds() / 3600  # hours
            })
        
        return pd.DataFrame(results)
    
    def compute_all_satellites(self, nav_df: pd.DataFrame, 
                             timestamps: List[datetime]) -> pd.DataFrame:
        """
        Compute broadcast positions for all satellites at given timestamps
        
        Args:
            nav_df: Navigation data DataFrame
            timestamps: List of target timestamps
            
        Returns:
            DataFrame with all computed positions
        """
        all_results = []
        
        # Get unique satellites
        satellites = nav_df['satellite'].unique()
        gps_satellites = [sat for sat in satellites if sat.startswith('G')]
        
        print(f"Computing broadcast positions for {len(gps_satellites)} GPS satellites at {len(timestamps)} timestamps")
        
        for i, satellite in enumerate(gps_satellites):
            print(f"  Processing {satellite} ({i+1}/{len(gps_satellites)})")
            sat_results = self.compute_positions_for_timestamps(nav_df, timestamps, satellite)
            if not sat_results.empty:
                all_results.append(sat_results)
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_results = combined_results.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            print(f"Successfully computed {len(combined_results)} broadcast positions")
            return combined_results
        else:
            return pd.DataFrame()


def create_simplified_ephemeris(satellite: str, timestamp: datetime) -> Dict:
    """
    Create a simplified ephemeris record for testing
    Uses approximate GPS orbital parameters
    """
    # Approximate GPS orbital parameters
    a = 26560000.0  # Semi-major axis in meters (approximately)
    e = 0.01  # Typical GPS eccentricity
    i0 = math.radians(55.0)  # Typical GPS inclination
    
    # GPS time from epoch
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    gps_seconds = (timestamp - gps_epoch).total_seconds()
    
    return {
        'timestamp': timestamp,
        'satellite': satellite,
        'a': a,
        'e': e,
        'i0': i0,
        'omega0': 0.0,
        'omega': 0.0,
        'm0': 0.0,
        'delta_n': 0.0,
        'omega_dot': 0.0,
        'idot': 0.0,
        'cuc': 0.0,
        'cus': 0.0,
        'crc': 0.0,
        'crs': 0.0,
        'cic': 0.0,
        'cis': 0.0,
        'toe': gps_seconds,
        'toc': gps_seconds,
        'af0': 0.0,
        'af1': 0.0,
        'af2': 0.0
    }


def test_broadcast_computer():
    """Test function to verify broadcast orbit computation"""
    computer = BroadcastOrbitComputer()
    
    # Test with simplified ephemeris
    test_time = datetime(2025, 7, 6, 12, 0, 0)
    test_satellite = 'G01'
    
    # Create test ephemeris
    eph = create_simplified_ephemeris(test_satellite, test_time)
    
    print("Testing broadcast orbit computation...")
    print(f"Test satellite: {test_satellite}")
    print(f"Test time: {test_time}")
    
    # Test position computation
    position = computer.compute_satellite_position(eph, test_time)
    if position:
        print(f"Computed position: X={position[0]:.1f}, Y={position[1]:.1f}, Z={position[2]:.1f} meters")
        
        # Check if position is reasonable (GPS orbit radius ~26,560 km)
        radius = math.sqrt(sum(coord**2 for coord in position))
        print(f"Orbital radius: {radius/1000:.1f} km")
        
        if 20000000 < radius < 30000000:  # Reasonable GPS orbit range
            print("✓ Position computation appears reasonable")
        else:
            print("⚠ Position may be incorrect - outside expected GPS orbit range")
    else:
        print("✗ Position computation failed")
    
    # Test clock correction
    clock_corr = computer.compute_satellite_clock_correction(eph, test_time)
    print(f"Clock correction: {clock_corr*1e9:.2f} nanoseconds")
    
    # Test Kepler equation solver
    M = math.pi / 4  # 45 degrees
    e = 0.01
    E = computer.solve_kepler_equation(M, e)
    expected_E = M + e * math.sin(E)  # Should satisfy Kepler's equation
    error = abs(E - e * math.sin(E) - M)
    print(f"Kepler equation test - Error: {error:.2e} (should be < 1e-12)")
    
    if error < 1e-10:
        print("✓ Kepler equation solver working correctly")
    else:
        print("⚠ Kepler equation solver may have issues")


if __name__ == "__main__":
    test_broadcast_computer()