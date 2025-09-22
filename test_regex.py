#!/usr/bin/env python3
"""
Test different regex patterns for scientific notation
"""

import re

def test_regex_patterns():
    test_line = "8.300000000000E+01-3.181250000000E+01 4.553403953266E-09 1.888137676262E+00"
    
    patterns = [
        r'[+-]?\d+\.?\d*[ED][+-]?\d+',
        r'[+-]?\d+\.\d*[ED][+-]?\d+',
        r'[-+]?\d+(?:\.\d*)?[ED][-+]?\d+',
        r'[-+]?\d+\.?\d*[ED][-+]?\d+'
    ]
    
    print(f"Test line: {test_line}")
    print()
    
    for i, pattern in enumerate(patterns):
        print(f"Pattern {i+1}: {pattern}")
        matches = re.findall(pattern, test_line)
        print(f"Matches: {matches}")
        print(f"Count: {len(matches)}")
        print()

if __name__ == "__main__":
    test_regex_patterns()