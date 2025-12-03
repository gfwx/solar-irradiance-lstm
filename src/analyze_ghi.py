import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor

def analyze_ghi():
    data_dir = "/Users/sk/Desktop/proj"
    processor = DataProcessor(data_dir=data_dir)
    
    print("Loading data...")
    df = processor.load_data()
    
    # Old sites (from previous run)
    old_sites = ['CAB', 'SEL', 'SYO', 'TAT']
    
    # Filter for old sites
    df_old = df[df['site'].isin(old_sites)]
    
    print("\n--- Statistics Analysis ---")
    
    mean_old = df_old[processor.target_column].mean()
    mean_all = df[processor.target_column].mean()
    
    print(f"Mean GHI (Old 4 Sites): {mean_old:.2f} W/m^2")
    print(f"Mean GHI (All Sites):   {mean_all:.2f} W/m^2")
    
    ratio = mean_old / mean_all
    print(f"Ratio (Old/All): {ratio:.2f}")
    
    if mean_all < mean_old:
        print("CONCLUSION: Mean GHI decreased, which explains the nRMSE increase (nRMSE = RMSE / Mean).")
    else:
        print("CONCLUSION: Mean GHI did not decrease. nRMSE increase is due to higher RMSE.")

    # Check for zeros
    zeros_old = (df_old[processor.target_column] == 0).sum() / len(df_old)
    zeros_all = (df[processor.target_column] == 0).sum() / len(df)
    
    print(f"\nZero GHI Fraction (Old): {zeros_old:.2%}")
    print(f"Zero GHI Fraction (All): {zeros_all:.2%}")

if __name__ == "__main__":
    analyze_ghi()
