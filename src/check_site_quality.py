import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor

def check_quality():
    data_dir = "/Users/sk/Desktop/proj"
    processor = DataProcessor(data_dir=data_dir)
    
    print("Loading data...")
    df = processor.load_data()
    
    print(f"\n{'Site':<10} | {'Temp Mean':<10} | {'Hum Mean':<10} | {'Press Mean':<10} | {'Status'}")
    print("-" * 65)
    
    valid_sites = []
    
    for site, site_df in df.groupby('site'):
        # Check for columns
        cols = site_df.columns
        has_temp = any('T2' in c for c in cols)
        has_hum = any('RH' in c for c in cols)
        has_press = any('Po' in c or 'pressure' in c.lower() for c in cols)
        
        # Check for zeros (if column exists)
        # We need to map them first to check values
        temp_col = next((c for c in cols if 'T2' in c), None)
        hum_col = next((c for c in cols if 'RH' in c), None)
        press_col = next((c for c in cols if 'Po' in c or 'pressure' in c.lower()), None)
        
        t_mean = site_df[temp_col].mean() if temp_col else 0.0
        h_mean = site_df[hum_col].mean() if hum_col else 0.0
        p_mean = site_df[press_col].mean() if press_col else 0.0
        
        status = "OK"
        if not (has_temp and has_hum and has_press):
            status = "MISSING COLS"
        elif t_mean == 0 and h_mean == 0: # Unlikely to have 0 mean temp AND humidity
            status = "ZERO FILLED"
        elif p_mean < 100: # Pressure usually > 800 hPa
            status = "BAD PRESSURE"
            
        print(f"{site:<10} | {t_mean:<10.2f} | {h_mean:<10.2f} | {p_mean:<10.2f} | {status}")
        
        if status == "OK":
            valid_sites.append(site)
            
    print("-" * 65)
    print(f"Valid Sites: {valid_sites}")
    print(f"Count: {len(valid_sites)}")

if __name__ == "__main__":
    check_quality()
