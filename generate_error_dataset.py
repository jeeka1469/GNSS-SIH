
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
from typing import Dict, List, Optional

from sp3_parser import SP3Parser
from clk_parser import CLKParser
from rinex_parser_fixed import RINEXParser
from broadcast_orbit import BroadcastOrbitComputer
from error_calculator import ErrorCalculator


class GNSSErrorDatasetGenerator:    
    def __init__(self, dataset_dir: str = "dataset", validation_dir: str = "validation"):
        self.dataset_dir = dataset_dir
        self.validation_dir = validation_dir
        
        self.sp3_parser = SP3Parser(dataset_dir)
        self.clk_parser = CLKParser(dataset_dir)
        self.rinex_parser = RINEXParser()
        self.orbit_computer = BroadcastOrbitComputer()
        self.error_calculator = ErrorCalculator()
        
        self.precise_orbit_data = None
        self.precise_clock_data = None
        self.navigation_data = None
        self.broadcast_positions = None
        self.error_data = None
        
    def step1_parse_precise_orbit_data(self) -> bool:
        print("=" * 60)
        print("STEP 1: Parsing SP3 files for precise orbit data")
        print("=" * 60)
        
        try:
            orbit_df, _ = self.sp3_parser.process_all_sp3_files()
            
            if orbit_df.empty:
                print("âŒ No orbit data found in SP3 files")
                return False
            
            self.precise_orbit_data = orbit_df
            print(f"âœ… Successfully parsed precise orbit data: {len(orbit_df)} records")
            print(f"   Time range: {orbit_df['timestamp'].min()} to {orbit_df['timestamp'].max()}")
            print(f"   Satellites: {orbit_df['satellite'].nunique()} unique satellites")
            
            return True
            
        except Exception as e:
            print(f"âŒ Error parsing SP3 files: {e}")
            return False
    
    def step2_parse_precise_clock_data(self) -> bool:
        print("\\n" + "=" * 60)
        print("STEP 2: Parsing CLK files for precise clock data")
        print("=" * 60)
        
        try:
            clock_df = self.clk_parser.process_all_clk_files()
            
            if clock_df.empty:
                print("âŒ No clock data found in CLK files")
                return False
            
            satellite_clocks = self.clk_parser.filter_satellite_clocks(clock_df)
            
            if satellite_clocks.empty:
                print("âŒ No satellite clock data found")
                return False
            
            self.precise_clock_data = satellite_clocks
            print(f"Successfully parsed precise clock data: {len(satellite_clocks)} records")
            print(f"   Time range: {satellite_clocks['timestamp'].min()} to {satellite_clocks['timestamp'].max()}")
            print(f"   Satellites: {satellite_clocks['satellite'].nunique()} unique satellites")
            
            return True
            
        except Exception as e:
            print(f"Error parsing CLK files: {e}")
            return False
    
    def step3_parse_navigation_data(self) -> bool:
        print("\\n" + "=" * 60)
        print("STEP 3: Parsing RINEX files for navigation data")
        print("=" * 60)
        
        try:
            rinex_files = []
            for file in os.listdir(self.validation_dir):
                if file.endswith('.rnx'):
                    file_path = os.path.join(self.validation_dir, file, file)
                    if os.path.exists(file_path):
                        rinex_files.append(file_path)
            
            if not rinex_files:
                print("âŒ No RINEX files found")
                return False
            
            print(f"Found {len(rinex_files)} RINEX files to process")
            
            all_nav_data = []
            for rinex_file in rinex_files:
                print(f"Parsing RINEX file: {os.path.basename(rinex_file)}")
                nav_df = self.rinex_parser.parse_file(rinex_file)
                if not nav_df.empty:
                    all_nav_data.append(nav_df)
                    print(f"  Extracted {len(nav_df)} navigation records")
            
            if not all_nav_data:
                print("âŒ No navigation data extracted from RINEX files")
                return False
            
            combined_nav = pd.concat(all_nav_data, ignore_index=True)
            combined_nav = combined_nav.sort_values(['timestamp', 'satellite']).reset_index(drop=True)
            
            gps_nav = combined_nav[combined_nav['satellite'].str.startswith('G')]
            
            if gps_nav.empty:
                print("âŒ No GPS navigation data found")
                return False
            
            self.navigation_data = gps_nav
            print(f"âœ… Successfully parsed navigation data: {len(gps_nav)} records")
            print(f"   Time range: {gps_nav['timestamp'].min()} to {gps_nav['timestamp'].max()}")
            print(f"   GPS Satellites: {gps_nav['satellite'].nunique()} unique satellites")
            
            return True
            
        except Exception as e:
            print(f"âŒ Error parsing RINEX files: {e}")
            return False
    
    def step4_compute_broadcast_positions(self) -> bool:
        print("\\n" + "=" * 60)
        print("STEP 4: Computing broadcast positions")
        print("=" * 60)
        
        try:
            if self.precise_orbit_data is None or self.navigation_data is None:
                print("âŒ Precise orbit data or navigation data not available")
                return False
            
            timestamps = sorted(self.precise_orbit_data['timestamp'].unique())
            print(f"Computing broadcast positions for {len(timestamps)} timestamps")
            
            broadcast_df = self.orbit_computer.compute_all_satellites(
                self.navigation_data, timestamps
            )
            
            if broadcast_df.empty:
                print("âŒ No broadcast positions computed")
                return False
            
            self.broadcast_positions = broadcast_df
            print(f"âœ… Successfully computed broadcast positions: {len(broadcast_df)} records")
            print(f"   Satellites: {broadcast_df['satellite'].nunique()} unique satellites")
            
            return True
            
        except Exception as e:
            print(f"âŒ Error computing broadcast positions: {e}")
            return False
    
    def step5_compute_errors(self) -> bool:
        print("\\n" + "=" * 60)
        print("STEP 5: Computing orbit and clock errors")
        print("=" * 60)
        
        try:
            if any(data is None for data in [self.broadcast_positions, 
                                           self.precise_orbit_data, 
                                           self.precise_clock_data]):
                print("âŒ Required data not available for error computation")
                return False
            
            error_df = self.error_calculator.compute_all_errors(
                self.broadcast_positions,
                self.precise_orbit_data,
                self.precise_clock_data
            )
            
            if error_df.empty:
                print("âŒ No errors computed")
                return False
            
            self.error_data = error_df
            print(f"âœ… Successfully computed errors: {len(error_df)} records")
            print(f"   Satellites: {error_df['satellite'].nunique()} unique satellites")
            
            stats = self.error_calculator.generate_error_statistics(error_df)
            
            if 'orbit_error' in stats:
                orbit_stats = stats['orbit_error']
                print(f"   Orbit Error Stats (meters):")
                print(f"     Mean: {orbit_stats['mean']:.2f}")
                print(f"     Std:  {orbit_stats['std']:.2f}")
                print(f"     95th percentile: {orbit_stats['percentile_95']:.2f}")
            
            if 'clock_error' in stats:
                clock_stats = stats['clock_error']
                print(f"   Clock Error Stats (nanoseconds):")
                print(f"     Mean: {clock_stats['mean']:.2f}")
                print(f"     Std:  {clock_stats['std']:.2f}")
                print(f"     95th percentile: {clock_stats['percentile_95']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"âŒ Error computing errors: {e}")
            return False
    
    def step6_save_dataset(self, output_filename: str = "errors_day187_193.csv") -> bool:
        """
        Step 6: Save the final error dataset to CSV
        
        Args:
            output_filename: Name of the output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        print("\\n" + "=" * 60)
        print("STEP 6: Saving final dataset")
        print("=" * 60)
        
        try:
            if self.error_data is None:
                print("âŒ No error data available to save")
                return False
            
            output_df = pd.DataFrame({
                'satellite_id': self.error_data['satellite'],
                'timestamp': self.error_data['timestamp'],
                'orbit_error_m': self.error_data['orbit_error_m'],
                'clock_error_ns': self.error_data['clock_error_ns']
            })
            
            if 'radial_error_m' in self.error_data.columns:
                output_df['radial_error_m'] = self.error_data['radial_error_m']
            if 'ephemeris_age_hours' in self.error_data.columns:
                output_df['ephemeris_age_hours'] = self.error_data['ephemeris_age_hours']
            
            output_df.to_csv(output_filename, index=False)
            
            print(f"âœ… Successfully saved dataset to {output_filename}")
            print(f"   Total records: {len(output_df)}")
            print(f"   Columns: {list(output_df.columns)}")
            
            stats_filename = output_filename.replace('.csv', '_statistics.json')
            if self.error_data is not None:
                stats = self.error_calculator.generate_error_statistics(self.error_data)
                with open(stats_filename, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                print(f"âœ… Saved statistics to {stats_filename}")
            
            return True
            
        except Exception as e:
            print(f"âŒ Error saving dataset: {e}")
            return False
    
    def run_complete_pipeline(self, output_filename: str = "errors_day187_193.csv") -> bool:
        """
        Run the complete data processing pipeline
        
        Args:
            output_filename: Name of the output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        print("ðŸš€ Starting GNSS Error Dataset Generation Pipeline")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Validation directory: {self.validation_dir}")
        print(f"Output file: {output_filename}")
        
        steps = [
            ("Parse SP3 orbit data", self.step1_parse_precise_orbit_data),
            ("Parse CLK clock data", self.step2_parse_precise_clock_data),
            ("Parse RINEX navigation", self.step3_parse_navigation_data),
            ("Compute broadcast positions", self.step4_compute_broadcast_positions),
            ("Compute errors", self.step5_compute_errors),
            ("Save dataset", lambda: self.step6_save_dataset(output_filename))
        ]
        
        start_time = datetime.now()
        
        for step_name, step_func in steps:
            step_start = datetime.now()
            print(f"\\nâ±ï¸  {step_name}...")
            
            success = step_func()
            
            if not success:
                print(f"\\nâŒ Pipeline failed at step: {step_name}")
                return False
            
            step_duration = (datetime.now() - step_start).total_seconds()
            print(f"   â±ï¸  Completed in {step_duration:.1f} seconds")
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        print("\\n" + "=" * 60)
        print("ðŸŽ‰ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total processing time: {total_duration:.1f} seconds")
        print(f"Output file: {output_filename}")
        
        if self.error_data is not None:
            print(f"Final dataset: {len(self.error_data)} error records")
            print(f"Date range: {self.error_data['timestamp'].min()} to {self.error_data['timestamp'].max()}")
            print(f"Satellites: {self.error_data['satellite'].nunique()} GPS satellites")
        
        return True


def main():
    """Main function to run the complete pipeline"""
    if not os.path.exists("dataset"):
        print("âŒ Dataset directory not found. Please ensure 'dataset' folder exists with SP3 and CLK files.")
        return
    
    if not os.path.exists("validation"):
        print("âŒ Validation directory not found. Please ensure 'validation' folder exists with RINEX files.")
        return
    
    generator = GNSSErrorDatasetGenerator()
    
    success = generator.run_complete_pipeline("errors_day187_192.csv")
    
    if success:
        print("\\nâœ… Success! Error dataset has been generated.")
        print("ðŸ“Š You can now analyze the errors_day187_192.csv file for orbit and clock error patterns.")
    else:
        print("\\nâŒ Pipeline failed. Please check error messages above.")


if __name__ == "__main__":
    main()
