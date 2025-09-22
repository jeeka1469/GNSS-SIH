"""
CLK Parser Module for GNSS Error Analysis
Parser Team Implementation

This module handles parsing of CLK (RINEX Clock) files to extract precise
satellite and station clock corrections with high temporal resolution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from typing import Dict, List, Tuple, Optional


class CLKParser:
    """Parser for RINEX Clock files (.CLK format)"""
    
    def __init__(self, data_dir: str = "dataset"):
        """Initialize CLK parser with data directory"""
        self.data_dir = data_dir
        self.clock_data = {}
        
    def find_clk_files(self) -> List[str]:
        """Find all CLK files in the dataset directory"""
        pattern = os.path.join(self.data_dir, "**/IGS*CLK.CLK", "*.CLK")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)
    
    def parse_clk_header(self, file_path: str) -> Dict:
        """Parse CLK file header to extract metadata"""
        header_info = {}
        stations = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if 'RINEX VERSION' in line:
                    header_info['version'] = line[:20].strip()
                    header_info['file_type'] = line[20:40].strip()
                    
                elif 'PGM / RUN BY / DATE' in line:
                    header_info['program'] = line[:20].strip()
                    header_info['run_by'] = line[20:40].strip()
                    header_info['date'] = line[40:60].strip()
                    
                elif 'LEAP SECONDS' in line:
                    header_info['leap_seconds'] = int(line[:6].strip())
                    
                elif '# / TYPES OF DATA' in line:
                    parts = line.split()
                    header_info['num_data_types'] = int(parts[0])
                    header_info['data_types'] = parts[1:]
                    
                elif 'ANALYSIS CENTER' in line:
                    header_info['analysis_center'] = line[:60].strip()
                    
                elif '# OF SOLN STA / TRF' in line:
                    parts = line.split()
                    header_info['num_stations'] = int(parts[0])
                    header_info['reference_frame'] = ' '.join(parts[1:]).split('/')[0].strip()
                    
                elif 'SOLN STA NAME / NUM' in line:
                    station_info = {
                        'name': line[:4].strip(),
                        'domes': line[5:14].strip(),
                        'x': float(line[15:29].strip()) if line[15:29].strip() else 0,
                        'y': float(line[30:44].strip()) if line[30:44].strip() else 0,
                        'z': float(line[45:59].strip()) if line[45:59].strip() else 0
                    }
                    stations.append(station_info)
                    
                elif 'END OF HEADER' in line:
                    break
        
        header_info['stations'] = stations
        return header_info
    
    def parse_clk_data(self, file_path: str) -> pd.DataFrame:
        """
        Parse CLK file data to extract clock corrections
        
        Returns:
            DataFrame with columns [timestamp, receiver/satellite, clock_bias, std_dev]
        """
        print(f"Parsing CLK file: {os.path.basename(file_path)}")
        
        clock_records = []
        header_end = False
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not header_end:
                    if 'END OF HEADER' in line:
                        header_end = True
                    continue
                
                if line.startswith('AS ') or line.startswith('AR '):
                    # AS = satellite clock, AR = receiver clock
                    parts = line.split()
                    
                    try:
                        record_type = parts[0]  # AS or AR
                        site_or_sat = parts[1]  # Site name or satellite ID
                        
                        # Parse timestamp
                        year = int(parts[2])
                        month = int(parts[3])
                        day = int(parts[4])
                        hour = int(parts[5])
                        minute = int(parts[6])
                        second = float(parts[7])
                        
                        timestamp = datetime(year, month, day, hour, minute, int(second))
                        if second != int(second):  # Handle fractional seconds
                            timestamp += timedelta(microseconds=int((second - int(second)) * 1e6))
                        
                        # Parse clock data
                        num_values = int(parts[8])
                        clock_bias = float(parts[9]) if len(parts) > 9 else 0.0
                        std_dev = float(parts[10]) if len(parts) > 10 else 0.0
                        
                        clock_records.append({
                            'timestamp': timestamp,
                            'type': record_type,
                            'id': site_or_sat,
                            'clock_bias': clock_bias,  # in seconds
                            'std_dev': std_dev,
                            'num_values': num_values
                        })
                        
                    except (ValueError, IndexError) as e:
                        # Skip invalid lines
                        continue
        
        if clock_records:
            df = pd.DataFrame(clock_records)
            df = df.sort_values(['timestamp', 'id']).reset_index(drop=True)
            print(f"  Extracted {len(df)} clock records")
            return df
        else:
            print("  No clock records found")
            return pd.DataFrame()
    
    def process_all_clk_files(self) -> pd.DataFrame:
        """Process all CLK files and combine into unified DataFrame"""
        all_clock_data = []
        
        clk_files = self.find_clk_files()
        print(f"Found {len(clk_files)} CLK files to process")
        
        for file_path in clk_files:
            try:
                clock_df = self.parse_clk_data(file_path)
                if not clock_df.empty:
                    all_clock_data.append(clock_df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Combine all data
        if all_clock_data:
            combined_clock = pd.concat(all_clock_data, ignore_index=True)
            combined_clock = combined_clock.sort_values(['timestamp', 'id']).reset_index(drop=True)
            print(f"Total combined: {len(combined_clock)} clock records")
            return combined_clock
        else:
            return pd.DataFrame()
    
    def filter_satellite_clocks(self, clock_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only satellite clock records (AS type)"""
        if clock_df.empty:
            return clock_df
            
        satellite_clocks = clock_df[clock_df['type'] == 'AS'].copy()
        satellite_clocks = satellite_clocks.drop('type', axis=1)  # Remove type column
        satellite_clocks = satellite_clocks.rename(columns={'id': 'satellite'})
        
        return satellite_clocks.reset_index(drop=True)
    
    def filter_station_clocks(self, clock_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only station/receiver clock records (AR type)"""
        if clock_df.empty:
            return clock_df
            
        station_clocks = clock_df[clock_df['type'] == 'AR'].copy()
        station_clocks = station_clocks.drop('type', axis=1)  # Remove type column
        station_clocks = station_clocks.rename(columns={'id': 'station'})
        
        return station_clocks.reset_index(drop=True)
    
    def interpolate_to_timestamps(self, clock_df: pd.DataFrame, 
                                 target_timestamps: List[datetime], 
                                 satellite: str) -> pd.DataFrame:
        """
        Interpolate clock data to match target timestamps for a specific satellite
        
        Args:
            clock_df: Clock data DataFrame
            target_timestamps: List of datetime objects to interpolate to
            satellite: Satellite ID (e.g., 'G01')
            
        Returns:
            DataFrame with interpolated clock values at target timestamps
        """
        if clock_df.empty:
            return pd.DataFrame()
            
        # Filter for specific satellite
        sat_data = clock_df[clock_df['satellite'] == satellite].copy()
        if sat_data.empty:
            return pd.DataFrame()
            
        sat_data = sat_data.sort_values('timestamp')
        
        # Convert timestamps to numeric for interpolation
        time_numeric = np.array([(t - sat_data['timestamp'].iloc[0]).total_seconds() 
                                for t in sat_data['timestamp']])
        target_numeric = np.array([(t - sat_data['timestamp'].iloc[0]).total_seconds() 
                                  for t in target_timestamps])
        
        # Interpolate clock bias
        interpolated_bias = np.interp(target_numeric, time_numeric, sat_data['clock_bias'])
        
        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': target_timestamps,
            'satellite': satellite,
            'clock_bias': interpolated_bias
        })
        
        return result
    
    def get_clock_at_epochs(self, clock_df: pd.DataFrame, 
                           epoch_timestamps: List[datetime]) -> pd.DataFrame:
        """
        Extract clock data at specific epochs for all satellites
        
        Args:
            clock_df: Combined clock data
            epoch_timestamps: List of target timestamps
            
        Returns:
            DataFrame with clock data at requested epochs
        """
        if clock_df.empty:
            return pd.DataFrame()
            
        satellite_clocks = self.filter_satellite_clocks(clock_df)
        
        epoch_data = []
        satellites = satellite_clocks['satellite'].unique()
        
        for satellite in satellites:
            interpolated = self.interpolate_to_timestamps(
                satellite_clocks, epoch_timestamps, satellite
            )
            epoch_data.append(interpolated)
        
        if epoch_data:
            result = pd.concat(epoch_data, ignore_index=True)
            return result.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
        else:
            return pd.DataFrame()


def test_clk_parser():
    """Test function to verify CLK parser functionality"""
    parser = CLKParser()
    
    # Test finding files
    files = parser.find_clk_files()
    print(f"Found CLK files: {len(files)}")
    for f in files[:3]:  # Show first 3
        print(f"  {os.path.basename(f)}")
    
    if files:
        # Test parsing single file
        print(f"\nTesting single file parsing...")
        clock_df = parser.parse_clk_data(files[0])
        print(f"Clock data shape: {clock_df.shape}")
        
        if not clock_df.empty:
            print(f"\nData types: {clock_df['type'].value_counts()}")
            
            # Show satellite clocks
            sat_clocks = parser.filter_satellite_clocks(clock_df)
            print(f"\nSatellite clocks shape: {sat_clocks.shape}")
            print("Sample satellite clock data:")
            print(sat_clocks.head())
            
            # Show station clocks
            sta_clocks = parser.filter_station_clocks(clock_df)
            print(f"\nStation clocks shape: {sta_clocks.shape}")
            if not sta_clocks.empty:
                print("Sample station clock data:")
                print(sta_clocks.head())


if __name__ == "__main__":
    test_clk_parser()