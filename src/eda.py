import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from src.data_processor import DataProcessor

def generate_eda_report(data_dir="/Users/sk/Desktop/proj"):
    print("Loading data for EDA...")
    processor = DataProcessor(data_dir)
    # Load raw-ish data (resampled but not normalized)
    df = processor.load_data()
    
    # 1. Basic Info
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Total Records: {len(df)}")
    print(f"Time Range: {df.index.min()} to {df.index.max()}")
    print(f"Number of Sites: {df['site'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Missing Values
    print("\n" + "="*50)
    print("MISSING VALUES (after resampling/filling)")
    print("="*50)
    print(df.isnull().sum())
    
    # 3. Site Summary
    print("\n" + "="*50)
    print("SITE SUMMARY")
    print("="*50)
    # Group by site
    site_stats = df.groupby('site').agg({
        'SWD [W/m**2]': ['count', 'mean', 'std', 'min', 'max'],
        'latitude': 'first',
        'longitude': 'first'
    })
    site_stats.columns = ['Count', 'Mean GHI', 'Std GHI', 'Min GHI', 'Max GHI', 'Lat', 'Lon']
    print(site_stats)
    
    # 4. Feature Engineering for Correlation
    print("\nGenerating derived features for correlation analysis...")
    df_eng = processor.feature_engineering(df.copy())
    
    print("\n" + "="*50)
    print("CORRELATION MATRIX (Top Features vs GHI)")
    print("="*50)
    # Select numeric columns
    numeric_df = df_eng.select_dtypes(include=['float64', 'int64'])
    if 'SWD [W/m**2]' in numeric_df.columns:
        corr = numeric_df.corr()['SWD [W/m**2]'].sort_values(ascending=False)
        print(corr)
    
    # 5. Visualizations
    os.makedirs("artifacts", exist_ok=True)
    
    # Plot 1: GHI Distribution per Site
    print("\nGenerating plots...")
    sites = df['site'].unique()
    n_sites = len(sites)
    rows = (n_sites + 3) // 4
    
    plt.figure(figsize=(20, 5 * rows))
    for i, site in enumerate(sites):
        plt.subplot(rows, 4, i+1)
        subset = df[df['site'] == site]
        plt.hist(subset['SWD [W/m**2]'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        lat = subset['latitude'].iloc[0]
        plt.title(f"{site} (Lat: {lat:.1f})")
        plt.xlabel("GHI [W/m^2]")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("artifacts/eda_ghi_distribution.png")
    print("Saved artifacts/eda_ghi_distribution.png")
    
    # Plot 2: Average Diurnal Cycle per Site
    plt.figure(figsize=(15, 10))
    df['hour'] = df.index.hour
    
    # Use a colormap
    cm = plt.get_cmap('tab20')
    
    for i, site in enumerate(sites):
        subset = df[df['site'] == site]
        # Group by hour and take mean
        daily_profile = subset.groupby('hour')['SWD [W/m**2]'].mean()
        lat = subset['latitude'].iloc[0]
        plt.plot(daily_profile.index, daily_profile.values, 
                 label=f"{site} ({lat:.1f})", 
                 linewidth=2, color=cm(i/n_sites))
    
    plt.title("Average Daily GHI Profile by Site (Diurnal Cycle)")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Mean GHI [W/m^2]")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("artifacts/eda_diurnal_cycle.png")
    print("Saved artifacts/eda_diurnal_cycle.png")

    # Plot 3: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select interesting columns
    cols_to_plot = ['SWD [W/m**2]', 'temperature', 'humidity', 'pressure', 
                    'latitude', 'longitude', 'hour_sin', 'day_of_year_sin', 
                    'GHI_lag_24h', 'GHI_lag_168h']
    
    # Filter only columns that exist
    cols_to_plot = [c for c in cols_to_plot if c in df_eng.columns]
    
    corr_mat = df_eng[cols_to_plot].corr()
    
    plt.imshow(corr_mat, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    tick_marks = np.arange(len(cols_to_plot))
    plt.xticks(tick_marks, cols_to_plot, rotation=45, ha='right')
    plt.yticks(tick_marks, cols_to_plot)
    
    # Add text annotations
    for i in range(len(cols_to_plot)):
        for j in range(len(cols_to_plot)):
            text = f"{corr_mat.iloc[i, j]:.2f}"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=8)
            
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("artifacts/eda_correlation.png")
    print("Saved artifacts/eda_correlation.png")

if __name__ == "__main__":
    generate_eda_report()
