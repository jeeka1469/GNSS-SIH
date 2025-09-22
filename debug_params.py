from rinex_parser_fixed import RINEXParser
from broadcast_orbit import BroadcastOrbitComputer
import pandas as pd

# Parse one RINEX file
parser = RINEXParser()
nav_df = parser.parse_file("validation/BRDC00IGS_R_20251870000_01D_MN.rnx/BRDC00IGS_R_20251870000_01D_MN.rnx")

print("RINEX Parser Output:")
print(f"Number of records: {len(nav_df)}")
print(f"Columns: {list(nav_df.columns)}")
print("\nSample record:")
sample_record = nav_df.iloc[0].to_dict()
for key, value in sample_record.items():
    print(f"  {key}: {value}")

print("\n" + "="*50)

# Check what broadcast orbit computer expects
orbit_computer = BroadcastOrbitComputer()
print("Testing broadcast orbit computation...")

# Try to compute position with sample record
try:
    result = orbit_computer.compute_satellite_position(sample_record, sample_record['timestamp'])
    print(f"✅ Computation successful: {result}")
except Exception as e:
    print(f"❌ Computation failed: {e}")
    
print("\n" + "="*50)

# Let's check the compute_satellite_position method signature
import inspect
sig = inspect.signature(orbit_computer.compute_satellite_position)
print(f"Method signature: {sig}")

# Let's see what parameters are actually required
print("\nLet's check the source code for required parameters...")