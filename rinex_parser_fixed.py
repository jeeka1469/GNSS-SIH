"""
RINEX Navigation Data Parser
Parses RINEX navigation files to extract broadcast ephemeris data
"""

import pandas as pd
from datetime import datetime
import re
import numpy as np
from typing import Dict, List, Optional, Tuple


class RINEXParser:
    """Parser for RINEX navigation files"""
    
    def __init__(self):
        self.version = None
        self.navigation_data = []
        
    def parse_file(self, filepath: str) -> pd.DataFrame:
        """Parse RINEX navigation file and return DataFrame"""
        try:
            # Try with georinex first
            try:
                import georinex as gr
                nav_data = gr.load(filepath)
                if nav_data is not None and not nav_data.empty:
                    return self._process_georinex_data(nav_data)
            except Exception as e:
                print(f"Georinex failed: {e}")
                print("Falling back to manual parsing...")
            
            # Fall back to manual parsing
            return self._parse_manually(filepath)
            
        except Exception as e:
            print(f"Error parsing RINEX file {filepath}: {e}")
            return pd.DataFrame()
    
    def _process_georinex_data(self, nav_data):
        """Process data from georinex"""
        nav_records = []
        
        if hasattr(nav_data, 'sv') and 'GPS' in nav_data.coords.get('sv', []):
            gps_data = nav_data.sel(sv=nav_data.sv.str.startswith('G'))
            
            for sv in gps_data.sv.values:
                for time in gps_data.time.values:
                    try:
                        sv_data = gps_data.sel(sv=sv, time=time)
                        
                        # Extract required parameters
                        sqrt_a = sv_data.get('sqrtA', np.nan)
                        if not np.isnan(sqrt_a):
                            record = {
                                'satellite': sv,
                                'timestamp': pd.to_datetime(time),
                                'epoch': pd.to_datetime(time),
                                'a': sqrt_a ** 2,  # Semi-major axis
                                'e': sv_data.get('Eccentricity', 0),
                                'i0': sv_data.get('Io', 0),
                                'omega0': sv_data.get('Omega0', 0),  # Capital Omega
                                'omega': sv_data.get('omega', 0),   # Lowercase omega 
                                'm0': sv_data.get('M0', 0),
                                'delta_n': sv_data.get('DeltaN', 0),
                                'omega_dot': sv_data.get('OmegaDot', 0),
                                'idot': sv_data.get('IDOT', 0),
                                'cuc': sv_data.get('Cuc', 0),
                                'cus': sv_data.get('Cus', 0),
                                'crc': sv_data.get('Crc', 0),
                                'crs': sv_data.get('Crs', 0),
                                'cic': sv_data.get('Cic', 0),
                                'cis': sv_data.get('Cis', 0),
                                'toe': sv_data.get('Toe', 0),
                                'af0': sv_data.get('SVclockBias', 0),
                                'af1': sv_data.get('SVclockDrift', 0),
                                'af2': sv_data.get('SVclockDriftRate', 0),
                            }
                            nav_records.append(record)
                    except Exception as e:
                        continue
        
        return pd.DataFrame(nav_records)
    
    def _parse_manually(self, filepath: str) -> pd.DataFrame:
        """Manual RINEX parsing with full parameter extraction"""
        nav_records = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        i = 0
        header_end = False
        
        while i < len(lines):
            line = lines[i]
            
            # Skip header
            if not header_end:
                if 'END OF HEADER' in line:
                    header_end = True
                i += 1
                continue
            
            # Look for GPS navigation records
            if len(line) >= 23 and line[0] == 'G':
                try:
                    # Parse satellite and time from first line
                    satellite = line[0:3].strip()
                    year = int(line[4:8])
                    month = int(line[9:11]) 
                    day = int(line[12:14])
                    hour = int(line[15:17])
                    minute = int(line[18:20])
                    second = float(line[21:23])
                    
                    # Create timestamp
                    timestamp = datetime(year, month, day, hour, minute, int(second))
                    
                    # Parse clock parameters from first line
                    af0 = self._parse_scientific_notation(line[23:42])
                    af1 = self._parse_scientific_notation(line[42:61])
                    af2 = self._parse_scientific_notation(line[61:80])
                    
                    # Parse the 7 broadcast orbit data lines
                    if i + 7 < len(lines):
                        orbit_params = []
                        for j in range(1, 8):
                            orbit_line = lines[i + j]
                            values = self._parse_orbit_line(orbit_line)
                            orbit_params.extend(values)
                        
                        if len(orbit_params) >= 26:  # Need at least 26 parameters
                            sqrt_a = orbit_params[7]  # Square root of semi-major axis
                            
                            record = {
                                'satellite': satellite,
                                'timestamp': timestamp,
                                'epoch': timestamp,
                                'a': sqrt_a ** 2,  # Semi-major axis (m²)
                                'e': orbit_params[5],  # Eccentricity
                                'i0': orbit_params[12],  # Inclination at reference time (rad)
                                'omega0': orbit_params[10],  # Right ascension of ascending node (rad)
                                'omega': orbit_params[14],  # Argument of perigee (rad)
                                'm0': orbit_params[3],  # Mean anomaly at reference time (rad)
                                'delta_n': orbit_params[2],  # Mean motion difference (rad/s)
                                'omega_dot': orbit_params[15],  # Rate of right ascension (rad/s)
                                'idot': orbit_params[16],  # Rate of inclination angle (rad/s)
                                'cuc': orbit_params[4],  # Amplitude of cosine harmonic correction to argument of latitude
                                'cus': orbit_params[6],  # Amplitude of sine harmonic correction to argument of latitude
                                'crc': orbit_params[13],  # Amplitude of cosine harmonic correction to orbit radius
                                'crs': orbit_params[1],  # Amplitude of sine harmonic correction to orbit radius
                                'cic': orbit_params[9],  # Amplitude of cosine harmonic correction to inclination
                                'cis': orbit_params[11],  # Amplitude of sine harmonic correction to inclination
                                'toe': orbit_params[8],  # Reference time of ephemeris (sec of GPS week)
                                'af0': af0,  # Clock bias (seconds)
                                'af1': af1,  # Clock drift (sec/sec)
                                'af2': af2,  # Clock drift rate (sec/sec²)
                                'iode': orbit_params[0],  # Issue of data (ephemeris)
                                'gps_week': orbit_params[18] if len(orbit_params) > 18 else 0,
                                'tgd': orbit_params[22] if len(orbit_params) > 22 else 0,
                                'iodc': orbit_params[23] if len(orbit_params) > 23 else 0,
                            }
                            nav_records.append(record)
                    
                    # Skip the next 7 lines (already processed)
                    i += 7
                        
                except (ValueError, IndexError) as e:
                    print(f"Error parsing navigation record: {e}")
                    
            i += 1
        
        if nav_records:
            df = pd.DataFrame(nav_records)
            # Filter GPS satellites only
            df = df[df['satellite'].str.startswith('G')]
            df = df.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            print(f"Successfully parsed {len(df)} navigation records")
            return df
        else:
            print("No navigation records found")
            return pd.DataFrame()
    
    def _parse_scientific_notation(self, value_str: str) -> float:
        """Parse scientific notation, handling both E and D formats"""
        try:
            value_str = value_str.strip().replace('D', 'E')
            return float(value_str)
        except ValueError:
            return 0.0
    
    def _parse_orbit_line(self, line: str) -> List[float]:
        """Parse a line containing orbital parameters"""
        import re
        
        # First try to split on spaces and check if all parts are valid
        parts = line.strip().split()
        if len(parts) >= 3:
            # Check if all parts are valid scientific notation
            all_valid = True
            for part in parts:
                # A valid scientific notation should match this pattern
                if not re.match(r'^[+-]?\d+\.?\d*[ED][+-]?\d+$', part):
                    all_valid = False
                    break
            
            if all_valid:
                values = []
                for part in parts:
                    values.append(self._parse_scientific_notation(part))
                return values
        
        # For concatenated values or invalid space-split, use regex
        pattern = r'[+-]?\d+\.?\d*[ED][+-]?\d+'
        matches = re.findall(pattern, line)
        
        values = []
        for match in matches:
            values.append(self._parse_scientific_notation(match))
        
        return values
    
    def get_navigation_data(self) -> List[Dict]:
        """Return parsed navigation data"""
        return self.navigation_data


# Test the parser
if __name__ == "__main__":
    parser = RINEXParser()
    
    # Test with one file
    test_file = "validation/BRDC00IGS_R_20251870000_01D_MN.rnx/BRDC00IGS_R_20251870000_01D_MN.rnx"
    df = parser.parse_file(test_file)
    
    if not df.empty:
        print(f"Parsed {len(df)} records")
        print(f"Satellites: {df['satellite'].unique()}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Sample record keys: {list(df.columns)}")
        
        # Check for required parameters
        required = ['a', 'e', 'i0', 'omega0', 'omega', 'm0', 'delta_n', 'omega_dot', 'idot',
                   'cuc', 'cus', 'crc', 'crs', 'cic', 'cis', 'toe']
        missing = [param for param in required if param not in df.columns]
        if missing:
            print(f"Missing parameters: {missing}")
        else:
            print("✅ All required parameters present")
    else:
        print("❌ No data parsed")