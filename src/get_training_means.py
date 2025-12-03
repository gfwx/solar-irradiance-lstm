import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor

def get_means():
    data_dir = "/Users/sk/Desktop/proj"
    # Use original sites logic (already reverted in data_processor.py)
    processor = DataProcessor(data_dir=data_dir)
    df = processor.load_data()
    # Feature engineering filters for valid sites
    df = processor.feature_engineering(df)
    
    print("\n--- Training Data Means ---")
    print(f"Humidity: {df['humidity'].mean()}")
    print(f"Pressure: {df['pressure'].mean()}")
    print(f"Latitude: {df['latitude'].mean()}")
    print(f"Longitude: {df['longitude'].mean()}")

if __name__ == "__main__":
    get_means()
