#!/usr/bin/env python3
"""
Test script to debug RINEX parameter parsing by printing raw orbit parameters
"""

import os
from rinex_parser_fixed import RINEXParser

def test_parse_raw_params():
    # Get a single RINEX file
    rinex_file = "validation/BRDC00IGS_R_20251870000_01D_MN.rnx/BRDC00IGS_R_20251870000_01D_MN.rnx"
    
    if not os.path.exists(rinex_file):
        print(f"RINEX file not found: {rinex_file}")
        return
    
    print("🔍 RAW PARAMETER DEBUGGING")
    print("=" * 50)
    
    # Parse manually - examining the manual parsing method
    parser = RINEXParser()
    
    with open(rinex_file, 'r') as f:
        lines = f.readlines()
    
    # Find first navigation record
    header_end = False
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip header section
        if not header_end:
            if "END OF HEADER" in line:
                header_end = True
                print(f"Header ends at line {i}")
            continue
        
        # Look for GPS satellite record (after header)
        if len(line) >= 23 and line[0] == 'G' and line[1:3].isdigit():
            satellite = line[0:3].strip()
            print(f"Found GPS satellite: {satellite}")
            print(f"Line {i}: {line}")
            
            # Parse the 7 broadcast orbit data lines
            if i + 7 < len(lines):
                orbit_params = []
                for j in range(1, 8):
                    orbit_line = lines[i + j]
                    print(f"Line {i+j}: {orbit_line.strip()}")
                    
                    # Use the parser's parsing method
                    values = parser._parse_orbit_line(orbit_line)
                    print(f"  Raw line: '{orbit_line.strip()}'")
                    print(f"  Parsed values ({len(values)}): {values}")
                    orbit_params.extend(values)
                
                print(f"\n📊 All orbit parameters ({len(orbit_params)} total):")
                for idx, param in enumerate(orbit_params):
                    print(f"  [{idx:2d}]: {param:15.6e}")
                
                # Show key parameter positions based on GPS ICD
                print(f"\n🔍 Key GPS Parameters:")
                print(f"  IODE (index 0): {orbit_params[0] if len(orbit_params) > 0 else 'N/A'}")
                print(f"  sqrt(A) (index 7): {orbit_params[7] if len(orbit_params) > 7 else 'N/A'}")
                print(f"  Expected sqrt(A) squared: {orbit_params[7]**2 if len(orbit_params) > 7 else 'N/A'}")
                
                # Try different possible positions for sqrt(A)
                for idx in range(min(26, len(orbit_params))):
                    val = orbit_params[idx]
                    if 5000 < val < 7000:  # Typical sqrt(A) range for GPS satellites
                        print(f"  ⭐ Potential sqrt(A) at index {idx}: {val} -> a = {val**2:.3e}")
            
            break  # Only process first record for debugging
    
if __name__ == "__main__":
    test_parse_raw_params()