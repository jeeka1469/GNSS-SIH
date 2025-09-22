#!/usr/bin/env python3
"""
Detailed debugging script to diagnose RINEX parser semi-major axis calculation
"""

import os
import pandas as pd
from rinex_parser_fixed import RINEXParser

def main():
    # Get a single RINEX file for detailed debugging
    rinex_file = "validation/BRDC00IGS_R_20251870000_01D_MN.rnx/BRDC00IGS_R_20251870000_01D_MN.rnx"
    
    if not os.path.exists(rinex_file):
        print(f"RINEX file not found: {rinex_file}")
        return
    
    print("🔍 DETAILED RINEX DEBUGGING")
    print("=" * 50)
    
    # Initialize parser
    parser = RINEXParser()
    
    # Parse the file
    nav_data = parser.parse_file(rinex_file)
    
    if nav_data is not None and len(nav_data) > 0:
        print(f"📊 Total records parsed: {len(nav_data)}")
        print(f"📋 Columns: {list(nav_data.columns)}")
        
        # Check for zero or NaN semi-major axis values
        zero_a = nav_data[nav_data['a'] == 0.0]
        nan_a = nav_data[nav_data['a'].isna()]
        valid_a = nav_data[(nav_data['a'] > 0) & nav_data['a'].notna()]
        
        print(f"\n📈 Semi-major axis 'a' analysis:")
        print(f"  - Zero values: {len(zero_a)}")
        print(f"  - NaN values: {len(nan_a)}")
        print(f"  - Valid values: {len(valid_a)}")
        
        if len(valid_a) > 0:
            print(f"  - Valid 'a' range: {valid_a['a'].min():.2e} to {valid_a['a'].max():.2e}")
            print(f"  - Mean 'a': {valid_a['a'].mean():.2e}")
            
            # Show a sample valid record
            sample_valid = valid_a.iloc[0]
            print(f"\n✅ Sample VALID record:")
            for col in sample_valid.index:
                print(f"  {col}: {sample_valid[col]}")
        
        if len(zero_a) > 0:
            # Show a sample zero record
            sample_zero = zero_a.iloc[0]
            print(f"\n❌ Sample ZERO 'a' record:")
            for col in sample_zero.index:
                print(f"  {col}: {sample_zero[col]}")
        
        # Check if specific satellites have issues
        satellites = nav_data['satellite'].unique()
        print(f"\n🛰️ Satellites found: {sorted(satellites)}")
        
        for sat in sorted(satellites)[:5]:  # Check first 5 satellites
            sat_data = nav_data[nav_data['satellite'] == sat]
            valid_sat = sat_data[(sat_data['a'] > 0) & sat_data['a'].notna()]
            print(f"  {sat}: {len(sat_data)} total, {len(valid_sat)} valid 'a'")
    
    else:
        print("❌ No navigation data parsed!")

if __name__ == "__main__":
    main()