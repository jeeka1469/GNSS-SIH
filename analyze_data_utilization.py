import pandas as pd
import os
from glob import glob

print("ðŸ“Š DATA UTILIZATION ANALYSIS")
print("=" * 60)

print("ðŸ” SOURCE FILE ANALYSIS:")

sp3_files = glob("dataset/*/IGS0OPSFIN_*_15M_ORB.SP3")
print(f"\nðŸ“ SP3 FILES (Precise Orbits):")
for i, sp3_file in enumerate(sp3_files[:3]):  # Check first 3
    try:
        with open(sp3_file, 'r') as f:
            lines = f.readlines()
        pos_records = len([line for line in lines if line.startswith('P')])
        print(f"   File {i+1}: {os.path.basename(sp3_file)}")
        print(f"           Total lines: {len(lines):,}")
        print(f"           Position records: {pos_records:,}")
        print(f"           Expected: ~3,000 (15-min intervals)")
    except Exception as e:
        print(f"   Error reading {sp3_file}: {e}")

clk_files = glob("dataset/*/IGS0OPSFIN_*_30S_CLK.CLK")
print(f"\nðŸ• CLK FILES (Precise Clocks):")
for i, clk_file in enumerate(clk_files[:3]):  # Check first 3
    try:
        with open(clk_file, 'r') as f:
            lines = f.readlines()
        clock_records = len([line for line in lines if line.startswith(('AS', 'AR'))])
        print(f"   File {i+1}: {os.path.basename(clk_file)}")
        print(f"           Total lines: {len(lines):,}")
        print(f"           Clock records: {clock_records:,}")
        print(f"           Expected: ~15,000 (30-sec intervals)")
    except Exception as e:
        print(f"   Error reading {clk_file}: {e}")

rinex_files = glob("validation/*/BRDC00IGS_R_*_MN.rnx")
print(f"\nðŸ“¡ RINEX FILES (Broadcast Ephemeris):")
for i, rinex_file in enumerate(rinex_files[:3]):  # Check first 3
    try:
        with open(rinex_file, 'r') as f:
            lines = f.readlines()
        print(f"   File {i+1}: {os.path.basename(rinex_file)}")
        print(f"           Total lines: {len(lines):,}")
        print(f"           Expected: ~165,000 lines (broadcast nav data)")
    except Exception as e:
        print(f"   Error reading {rinex_file}: {e}")

print(f"\nðŸ“ˆ OUR DATASET UTILIZATION:")
df = pd.read_csv('errors_day187_192.csv')
print(f"   Total records in CSV: {len(df):,}")
print(f"   Records per day: {len(df) // 7:,}")
print(f"   Records per satellite per day: {len(df) // (7 * 32):,}")
print(f"   Time interval: 15 minutes")

print(f"\nðŸ’¡ POTENTIAL FOR HIGHER RESOLUTION:")
print(f"ðŸ“ SP3 Data: 15-minute intervals (~3K records/day)")
print(f"   â†’ We're using: 15-min intervals âœ… FULLY UTILIZED")
print(f"")
print(f"ðŸ• CLK Data: 30-second intervals (~15K records/day)")  
print(f"   â†’ We're using: 15-min intervals (interpolated)")
print(f"   â†’ Potential: Could use 30-sec resolution (30x more data!)")
print(f"")
print(f"ðŸ“¡ RINEX Data: Multiple ephemeris per satellite per day")
print(f"   â†’ We're using: Closest ephemeris for each timestamp")
print(f"   â†’ Potential: Full ephemeris history available")

print(f"\nðŸš€ OPPORTUNITIES FOR ENHANCEMENT:")
print(f"1. Higher temporal resolution: 30-sec instead of 15-min")
print(f"   â†’ Would give 30x more training data points!")
print(f"2. More precise interpolation of CLK data")
print(f"3. Ephemeris age tracking with full history")
print(f"4. Sub-minute error analysis capability")
