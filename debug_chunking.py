
from rinex_parser_fixed import RINEXParser

def debug_chunking():
    test_line = "8.300000000000E+01-3.181250000000E+01 4.553403953266E-09 1.888137676262E+00"
    parser = RINEXParser()
    
    print(f"Test line: {test_line}")
    print(f"Length: {len(test_line)}")
    print()
    
    # Pad to 76 characters as the method does
    padded_line = test_line.ljust(76)
    print(f"Padded line: '{padded_line}'")
    print(f"Padded length: {len(padded_line)}")
    print()
    
    # Test each 19-character chunk manually
    for i in range(4):
        start = i * 19
        end = start + 19
        chunk = padded_line[start:end]
        print(f"Chunk {i}: pos {start:2d}-{end:2d}: '{chunk}' -> {parser._parse_scientific_notation(chunk.strip())}")

if __name__ == "__main__":
    debug_chunking()