import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor

def debug_filtering():
    data_dir = "/Users/sk/Desktop/proj"
    processor = DataProcessor(data_dir=data_dir)
    
    print("Loading data...")
    df = processor.load_data()
    print(f"Initial sites: {df['site'].unique()}")
    
    print("\nRunning Feature Engineering (with filtering)...")
    df_fe = processor.feature_engineering(df)
    
    print(f"\nRemaining sites: {df_fe['site'].unique()}")
    
    # Check if bad sites are still there
    bad_sites = ['COC', 'PAY', 'QIQ', 'TAM', 'TOR']
    remaining_bad = [s for s in bad_sites if s in df_fe['site'].unique()]
    
    if remaining_bad:
        print(f"ERROR: Bad sites still present: {remaining_bad}")
    else:
        print("SUCCESS: Bad sites successfully filtered.")
        
    # Check for NaNs in remaining data
    print("\nChecking for NaNs in remaining data:")
    print(df_fe[['temperature', 'humidity', 'pressure']].isna().sum())
    
    # Check for Zeros in remaining data (which would indicate fillna(0) happened)
    print("\nChecking for Zeros in remaining data:")
    print((df_fe[['temperature', 'humidity', 'pressure']] == 0).sum())

if __name__ == "__main__":
    debug_filtering()
