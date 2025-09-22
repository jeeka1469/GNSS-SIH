import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from typing import Dict, List, Tuple, Optional


class SP3Parser:
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.orbit_data = {}
        self.clock_data = {}
        
    def find_sp3_files(self) -> List[str]:
        pattern = os.path.join(self.data_dir, "**/IGS*ORB.SP3", "*.SP3")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)
    
    def parse_sp3_header(self, file_path: str) -> Dict:
        header_info = {}
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('#c'):
                    parts = line.split()
                    header_info['version'] = parts[0][:2]
                    header_info['pos_vel_flag'] = parts[0][2]
                    header_info['start_year'] = int(parts[0][3:])
                    header_info['start_month'] = int(parts[1])
                    header_info['start_day'] = int(parts[2])
                    header_info['start_hour'] = int(parts[3])
                    header_info['start_minute'] = int(parts[4])
                    header_info['start_second'] = float(parts[5])
                    header_info['num_epochs'] = int(parts[6])
                    
                    try:
                        header_info['start_datetime'] = datetime(
                            header_info['start_year'],
                            header_info['start_month'],
                            header_info['start_day'],
                            header_info['start_hour'],
                            header_info['start_minute'],
                            int(header_info['start_second'])
                        )
                    except ValueError as e:
                        print(f"Error parsing start time: {e}")
                        header_info['start_datetime'] = None
                    
                    break
                    
                if i > 50:
                    break
        
        return header_info