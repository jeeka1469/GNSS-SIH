#!/usr/bin/env python3
"""
Test the orbit line parsing method specifically
"""

from rinex_parser_fixed import RINEXParser

def test_orbit_line_parsing():
    parser = RINEXParser()
    
    # Test lines from the debug output (both spaced and concatenated)
    test_lines = [
        "-1.793727278709E-06 5.913949571550E-04 2.250075340271E-06 5.153716270447E+03",
        "8.300000000000E+01-3.181250000000E+01 4.553403953266E-09 1.888137676262E+00"
    ]
    
    print("🔍 ORBIT LINE PARSING TEST")
    print("=" * 50)
    
    for i, test_line in enumerate(test_lines):
        print(f"\nTest line {i+1}: {test_line}")
        print(f"Line length: {len(test_line)}")
        
        # Test the parsing method
        values = parser._parse_orbit_line(test_line)
        print(f"Parsed values: {values}")
        print(f"Number of values: {len(values)}")
        
        # Show expected sqrt(A) if this is the right line
        if len(values) >= 4 and abs(values[3]) > 1000:
            sqrt_a = values[3]
            a = sqrt_a ** 2
            print(f"⭐ Potential sqrt(A) = {sqrt_a}, A = {a:.3e}")
    
if __name__ == "__main__":
    test_orbit_line_parsing()