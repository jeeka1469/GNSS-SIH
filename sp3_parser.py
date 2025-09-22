import pandas as pd    def parse_sp3_header(self, file_path: str) -> Dict:mport numpy as np
from datetime import datetime, timedelta
import os
import glob
from typing import Dict, List, Tuple, Optional


class SP3Parser:
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.orbit_data = {}
        self.clock_data = {}
        
    def find_sp3_files(self) -> List[str]:
        pattern = os.path.join(self.data_dir, "**/IGS*ORB.SP3", "*.SP3")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)
    
    def parse_sp3_header(self, file_path: str) -> Dict:
        """Parse SP3 file header to extract metadata"""
        header_info = {}
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('#c'):  # First line with version and epoch info
                    parts = line.split()
                    # SP3 format: #cP YYYY MM DD HH MM SS.SSSSSS NUM_EPOCHS
                    # But parts[0] contains #cPYYYY as one token
                    header_info['version'] = parts[0][:2]  # #c
                    header_info['pos_vel_flag'] = parts[0][2]  # P
                    header_info['start_year'] = int(parts[0][3:])  # 2025 from #cP2025
                    header_info['start_month'] = int(parts[1])     # 7
                    header_info['start_day'] = int(parts[2])       # 6  
                    header_info['start_hour'] = int(parts[3])      # 0
                    header_info['start_minute'] = int(parts[4])    # 0
                    header_info['start_second'] = float(parts[5])  # 0.00000000
                    header_info['num_epochs'] = int(parts[6])      # 96
                    
                elif line.startswith('##'):  # Second line with interval info
                    parts = line.split()
                    header_info['gps_week'] = int(parts[1])
                    header_info['seconds_of_week'] = float(parts[2])
                    header_info['epoch_interval'] = float(parts[3])
                    
                elif line.startswith('+   '):  # Satellite list
                    # Extract satellite IDs
                    sats = line[4:].strip().split()
                    if 'satellites' not in header_info:
                        header_info['satellites'] = []
                    header_info['satellites'].extend([sat for sat in sats if sat and sat != '0'])
                    
                elif line.startswith('*'):  # Start of data section
                    break
                    
        return header_info
    
    def parse_sp3_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse SP3 file data to extract orbit positions and clock corrections
        
        Returns:
            orbit_df: DataFrame with columns [timestamp, satellite, x, y, z]
            clock_df: DataFrame with columns [timestamp, satellite, clock_bias]
        """
        print(f"Parsing SP3 file: {os.path.basename(file_path)}")
        
        header_info = self.parse_sp3_header(file_path)
        
        orbit_records = []
        clock_records = []
        current_epoch = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('*'):  # Epoch line
                    parts = line.split()
                    year = int(parts[1])
                    month = int(parts[2])
                    day = int(parts[3])
                    hour = int(parts[4])
                    minute = int(parts[5])
                    second = float(parts[6])
                    
                    current_epoch = datetime(year, month, day, hour, minute, int(second))
                    if second != int(second):  # Handle fractional seconds
                        current_epoch += timedelta(microseconds=int((second - int(second)) * 1e6))
                
                elif line.startswith('P') and current_epoch:  # Position line
                    parts = line.split()
                    satellite = parts[0][1:]  # Remove 'P' prefix
                    
                    try:
                        x = float(parts[1]) * 1000  # Convert km to meters
                        y = float(parts[2]) * 1000
                        z = float(parts[3]) * 1000
                        clock = float(parts[4]) * 1e-6  # Convert microseconds to seconds
                        
                        # Store orbit data
                        orbit_records.append({
                            'timestamp': current_epoch,
                            'satellite': satellite,
                            'x': x,
                            'y': y,
                            'z': z
                        })
                        
                        # Store clock data (if not 999999.999999)
                        if abs(clock - 999999.999999e-6) > 1e-10:
                            clock_records.append({
                                'timestamp': current_epoch,
                                'satellite': satellite,
                                'clock_bias': clock
                            })
                            
                    except (ValueError, IndexError):
                        # Skip invalid lines
                        continue
        
        orbit_df = pd.DataFrame(orbit_records)
        clock_df = pd.DataFrame(clock_records)
        
        # Sort by timestamp and satellite
        if not orbit_df.empty:
            orbit_df = orbit_df.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
        if not clock_df.empty:
            clock_df = clock_df.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            
        print(f"  Extracted {len(orbit_df)} orbit records and {len(clock_df)} clock records")
        
        return orbit_df, clock_df
    
    def process_all_sp3_files(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all SP3 files and combine into unified DataFrames"""
        all_orbit_data = []
        all_clock_data = []
        
        sp3_files = self.find_sp3_files()
        print(f"Found {len(sp3_files)} SP3 files to process")
        
        for file_path in sp3_files:
            try:
                orbit_df, clock_df = self.parse_sp3_data(file_path)
                all_orbit_data.append(orbit_df)
                all_clock_data.append(clock_df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Combine all data
        if all_orbit_data:
            combined_orbit = pd.concat(all_orbit_data, ignore_index=True)
            combined_orbit = combined_orbit.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
        else:
            combined_orbit = pd.DataFrame()
            
        if all_clock_data:
            combined_clock = pd.concat(all_clock_data, ignore_index=True)
            combined_clock = combined_clock.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
        else:
            combined_clock = pd.DataFrame()
        
        print(f"Total combined: {len(combined_orbit)} orbit records, {len(combined_clock)} clock records")
        
        return combined_orbit, combined_clock
    
    def get_uniform_intervals(self, orbit_df: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
        """
        Extract data at uniform intervals (default 15 minutes to match SP3 native interval)
        """
        if orbit_df.empty:
            return orbit_df
            
        # Group by satellite
        uniform_data = []
        
        for satellite in orbit_df['satellite'].unique():
            sat_data = orbit_df[orbit_df['satellite'] == satellite].copy()
            sat_data = sat_data.sort_values('timestamp')
            
            # Create time grid at uniform intervals
            start_time = sat_data['timestamp'].min()
            end_time = sat_data['timestamp'].max()
            
            # Round start time to nearest interval
            start_rounded = start_time.replace(minute=(start_time.minute // interval_minutes) * interval_minutes, 
                                             second=0, microsecond=0)
            
            time_grid = []
            current_time = start_rounded
            while current_time <= end_time:
                time_grid.append(current_time)
                current_time += timedelta(minutes=interval_minutes)
            
            # Find closest matches for each time in grid
            for target_time in time_grid:
                # Find closest timestamp
                time_diffs = abs(sat_data['timestamp'] - target_time)
                closest_idx = time_diffs.idxmin()
                
                # Only include if within reasonable tolerance (e.g., 1 minute)
                if time_diffs[closest_idx] <= timedelta(minutes=1):
                    row = sat_data.loc[closest_idx].copy()
                    row['timestamp'] = target_time  # Use the grid time
                    uniform_data.append(row)
        
        if uniform_data:
            uniform_df = pd.DataFrame(uniform_data)
            uniform_df = uniform_df.drop_duplicates(['timestamp', 'satellite']).reset_index(drop=True)
            return uniform_df.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
        else:
            return pd.DataFrame()


def test_sp3_parser():
    """Test function to verify SP3 parser functionality"""
    parser = SP3Parser()
    
    # Test finding files
    files = parser.find_sp3_files()
    print(f"Found SP3 files: {len(files)}")
    for f in files[:3]:  # Show first 3
        print(f"  {os.path.basename(f)}")
    
    if files:
        # Test parsing single file
        print(f"\nTesting single file parsing...")
        orbit_df, clock_df = parser.parse_sp3_data(files[0])
        print(f"Sample orbit data shape: {orbit_df.shape}")
        print(f"Sample clock data shape: {clock_df.shape}")
        
        if not orbit_df.empty:
            print("\nSample orbit data:")
            print(orbit_df.head())
            
        if not clock_df.empty:
            print("\nSample clock data:")
            print(clock_df.head())


if __name__ == "__main__":
    test_sp3_parser()