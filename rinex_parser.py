"""
RINEX Navigation Parser Module for GNSS Error Analysis
Parser Team Implementation

This module handles parsing of RINEX navigation files to extract broadcast
ephemeris data for satellite position computation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from typing import Dict, List, Tuple, Optional
import georinex as gr


class RINEXParser:
    """Parser for RINEX Navigation files (.rnx format)"""
    
    def __init__(self, data_dir: str = "validation"):
        """Initialize RINEX parser with data directory"""
        self.data_dir = data_dir
        self.nav_data = {}
        
    def find_rinex_files(self) -> List[str]:
        """Find all RINEX navigation files in the validation directory"""
        pattern = os.path.join(self.data_dir, "**/BRDC*.rnx", "*.rnx")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)
    
    def parse_rinex_nav(self, file_path: str) -> pd.DataFrame:
        """
        Parse RINEX navigation file using georinex
        
        Returns:
            DataFrame with broadcast ephemeris parameters
        """
        print(f"Parsing RINEX file: {os.path.basename(file_path)}")
        
        try:
            # Use georinex to read the navigation file
            nav = gr.load(file_path)
            
            # Convert to DataFrame format suitable for orbit computation
            nav_records = []
            
            for sat in nav.sv.values:
                for time in nav.time.values:
                    # Check if data exists for this satellite at this time
                    sat_data = nav.sel(sv=sat, time=time, method='nearest')
                    
                    # Skip if essential parameters are missing
                    required_params = ['Toe', 'M0', 'DeltaN', 'Eccentricity', 'sqrtA', 
                                     'Omega0', 'i0', 'omega', 'OmegaDot', 'IDOT']
                    
                    if any(pd.isna(sat_data.get(param, np.nan)) for param in required_params):
                        continue
                    
                    # Convert time to datetime
                    timestamp = pd.to_datetime(time).to_pydatetime()
                    
                    record = {
                        'timestamp': timestamp,
                        'satellite': str(sat),
                        'toe': float(sat_data.get('Toe', 0)),           # Time of ephemeris
                        'toc': float(sat_data.get('Toc', 0)),           # Time of clock
                        'a': float(sat_data.get('sqrtA', 0))**2,        # Semi-major axis (m)
                        'e': float(sat_data.get('Eccentricity', 0)),    # Eccentricity
                        'i0': float(sat_data.get('i0', 0)),             # Inclination at reference time (rad)
                        'omega0': float(sat_data.get('Omega0', 0)),     # Longitude of ascending node (rad)
                        'omega': float(sat_data.get('omega', 0)),       # Argument of perigee (rad)
                        'm0': float(sat_data.get('M0', 0)),             # Mean anomaly at reference time (rad)
                        'delta_n': float(sat_data.get('DeltaN', 0)),    # Mean motion difference (rad/s)
                        'omega_dot': float(sat_data.get('OmegaDot', 0)), # Rate of right ascension (rad/s)
                        'idot': float(sat_data.get('IDOT', 0)),         # Rate of inclination angle (rad/s)
                        'cuc': float(sat_data.get('Cuc', 0)),           # Amplitude of cosine harmonic correction term to argument of latitude (rad)
                        'cus': float(sat_data.get('Cus', 0)),           # Amplitude of sine harmonic correction term to argument of latitude (rad)
                        'crc': float(sat_data.get('Crc', 0)),           # Amplitude of cosine harmonic correction term to orbit radius (m)
                        'crs': float(sat_data.get('Crs', 0)),           # Amplitude of sine harmonic correction term to orbit radius (m)
                        'cic': float(sat_data.get('Cic', 0)),           # Amplitude of cosine harmonic correction term to angle of inclination (rad)
                        'cis': float(sat_data.get('Cis', 0)),           # Amplitude of sine harmonic correction term to angle of inclination (rad)
                        'af0': float(sat_data.get('SVclockBias', 0)),   # Clock bias (s)
                        'af1': float(sat_data.get('SVclockDrift', 0)),  # Clock drift (s/s)
                        'af2': float(sat_data.get('SVclockDriftRate', 0)), # Clock drift rate (s/s²)
                        'iode': int(sat_data.get('IODE', 0)),           # Issue of data ephemeris
                        'iodc': int(sat_data.get('IODC', 0)),           # Issue of data clock
                        'week': int(sat_data.get('GPSWeek', 0)),        # GPS week
                        'health': int(sat_data.get('SVhealth', 0)),     # Satellite health
                        'tgd': float(sat_data.get('TGD', 0)),           # Group delay (s)
                        'fit_interval': float(sat_data.get('FitInterval', 4)) # Fit interval (hours)
                    }
                    
                    nav_records.append(record)
            
            if nav_records:
                df = pd.DataFrame(nav_records)
                df = df.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
                print(f"  Extracted {len(df)} navigation records")
                return df
            else:
                print("  No navigation records found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  Error parsing with georinex: {e}")
            # Fallback to manual parsing if georinex fails
            return self.parse_rinex_manual(file_path)
    
    def parse_rinex_manual(self, file_path: str) -> pd.DataFrame:
        """
        Fallback manual parser for RINEX navigation files
        """
        print(f"  Attempting manual parsing...")
        
        nav_records = []
        header_end = False
        
        with open(file_path, 'r') as f:
            for line in f:
                if not header_end:
                    if 'END OF HEADER' in line:
                        header_end = True
                    continue
                
                # Parse navigation data records
                if len(line) >= 23 and line[0] in ['G', 'R', 'E', 'C', 'J']:
                    try:
                        # First line of navigation record
                        satellite = line[0:3].strip()
                        year = int(line[4:8])
                        month = int(line[9:11])
                        day = int(line[12:14])
                        hour = int(line[15:17])
                        minute = int(line[18:20])
                        second = float(line[21:23])
                        
                        timestamp = datetime(year, month, day, hour, minute, int(second))
                        
                        # Parse clock parameters from first line
                        af0 = float(line[23:42].replace('D', 'E'))
                        af1 = float(line[42:61].replace('D', 'E'))
                        af2 = float(line[61:80].replace('D', 'E'))
                        
                        # Read the next 7 lines for orbital parameters
                        orbit_lines = []
                        for _ in range(7):
                            orbit_line = f.readline()
                            if orbit_line:
                                orbit_lines.append(orbit_line)
                        
                        if len(orbit_lines) == 7:
                            # Parse orbital parameters (simplified version)
                            record = {
                                'timestamp': timestamp,
                                'satellite': satellite,
                                'af0': af0,
                                'af1': af1,
                                'af2': af2,
                                # Add other parameters as needed...
                                # This is a simplified version
                            }
                            nav_records.append(record)
                            
                    except (ValueError, IndexError):
                        continue
        
        if nav_records:
            df = pd.DataFrame(nav_records)
            df = df.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            print(f"  Manually extracted {len(df)} navigation records")
            return df
        else:
            print("  Manual parsing found no records")
            return pd.DataFrame()
    
    def process_all_rinex_files(self) -> pd.DataFrame:
        """Process all RINEX navigation files and combine into unified DataFrame"""
        all_nav_data = []
        
        rinex_files = self.find_rinex_files()
        print(f"Found {len(rinex_files)} RINEX files to process")
        
        for file_path in rinex_files:
            try:
                nav_df = self.parse_rinex_nav(file_path)
                if not nav_df.empty:
                    all_nav_data.append(nav_df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Combine all data
        if all_nav_data:
            combined_nav = pd.concat(all_nav_data, ignore_index=True)
            combined_nav = combined_nav.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            print(f"Total combined: {len(combined_nav)} navigation records")
            return combined_nav
        else:
            return pd.DataFrame()
    
    def get_ephemeris_for_satellite(self, nav_df: pd.DataFrame, 
                                   satellite: str, 
                                   target_time: datetime) -> Optional[Dict]:
        """
        Get the most appropriate ephemeris record for a satellite at a given time
        
        Args:
            nav_df: Navigation data DataFrame
            satellite: Satellite ID (e.g., 'G01')
            target_time: Target datetime for ephemeris
            
        Returns:
            Dictionary with ephemeris parameters or None if not found
        """
        if nav_df.empty:
            return None
            
        # Filter for specific satellite
        sat_data = nav_df[nav_df['satellite'] == satellite].copy()
        if sat_data.empty:
            return None
        
        # Find the ephemeris with the closest time of ephemeris (toe)
        # that is not too far from the target time
        sat_data['time_diff'] = abs(sat_data['timestamp'] - target_time)
        
        # Filter to ephemeris that are within reasonable time range (e.g., 2 hours)
        max_age = timedelta(hours=2)
        valid_eph = sat_data[sat_data['time_diff'] <= max_age]
        
        if valid_eph.empty:
            return None
            
        # Return the closest one
        closest_idx = valid_eph['time_diff'].idxmin()
        return valid_eph.loc[closest_idx].to_dict()
    
    def filter_gps_satellites(self, nav_df: pd.DataFrame) -> pd.DataFrame:
        """Filter navigation data to only GPS satellites (G##)"""
        if nav_df.empty:
            return nav_df
            
        gps_data = nav_df[nav_df['satellite'].str.startswith('G')].copy()
        return gps_data.reset_index(drop=True)


def test_rinex_parser():
    """Test function to verify RINEX parser functionality"""
    parser = RINEXParser()
    
    # Test finding files
    files = parser.find_rinex_files()
    print(f"Found RINEX files: {len(files)}")
    for f in files[:3]:  # Show first 3
        print(f"  {os.path.basename(f)}")
    
    if files:
        # Test parsing single file
        print(f"\nTesting single file parsing...")
        nav_df = parser.parse_rinex_nav(files[0])
        print(f"Navigation data shape: {nav_df.shape}")
        
        if not nav_df.empty:
            print("\nAvailable columns:")
            print(nav_df.columns.tolist())
            
            print("\nSample navigation data:")
            print(nav_df.head())
            
            # Test GPS filtering
            gps_nav = parser.filter_gps_satellites(nav_df)
            print(f"\nGPS satellites: {gps_nav['satellite'].nunique()} unique satellites")
            print(f"GPS navigation records: {len(gps_nav)}")
            
            # Test ephemeris lookup
            if not gps_nav.empty:
                test_sat = gps_nav['satellite'].iloc[0]
                test_time = gps_nav['timestamp'].iloc[0]
                eph = parser.get_ephemeris_for_satellite(gps_nav, test_sat, test_time)
                if eph:
                    print(f"\nTest ephemeris lookup for {test_sat}:")
                    print(f"  Timestamp: {eph['timestamp']}")
                    print(f"  Semi-major axis: {eph.get('a', 'N/A')}")
                    print(f"  Eccentricity: {eph.get('e', 'N/A')}")


if __name__ == "__main__":
    test_rinex_parser()