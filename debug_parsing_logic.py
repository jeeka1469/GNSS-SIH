#!/usr/bin/env python3
"""
Debug the orbit line parsing logic step by step
"""

from rinex_parser_fixed import RINEXParser

def debug_parsing_logic():
    test_line = "8.300000000000E+01-3.181250000000E+01 4.553403953266E-09 1.888137676262E+00"
    parser = RINEXParser()
    
    print(f"Test line: {test_line}")
    print()
    
    # Test space splitting
    parts = test_line.strip().split()
    print(f"Space split parts: {parts}")
    print(f"Number of parts: {len(parts)}")
    print()
    
    if len(parts) >= 3:
        print("Taking space-split branch...")
        values = []
        for part in parts:
            if part:
                val = parser._parse_scientific_notation(part)
                print(f"  '{part}' -> {val}")
                values.append(val)
        print(f"Space-split result: {values}")
    else:
        print("Taking regex branch...")
        import re
        pattern = r'[+-]?\d+\.?\d*[ED][+-]?\d+'
        matches = re.findall(pattern, test_line)
        print(f"Regex matches: {matches}")
        
        values = []
        for match in matches:
            val = parser._parse_scientific_notation(match)
            print(f"  '{match}' -> {val}")
            values.append(val)
        print(f"Regex result: {values}")

if __name__ == "__main__":
    debug_parsing_logic()