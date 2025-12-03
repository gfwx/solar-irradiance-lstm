import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor

def check_counts():
    data_dir = "/Users/sk/Desktop/proj"
    
    # Check 24h Output
    print("Checking 24h Output Window...")
    p24 = DataProcessor(data_dir=data_dir, input_window=336, output_window=24)
    # We can skip full processing if we just want to count, but create_sequences needs data.
    # Let's just run it, it's fast enough.
    df = p24.load_data()
    df = p24.feature_engineering(df)
    df = p24.train_gbdt(df)
    df = p24.normalize(df)
    X24, _ = p24.create_sequences(df)
    print(f"Output 24h -> Samples: {len(X24)}")
    
    # Check 168h Output
    print("\nChecking 168h Output Window...")
    p168 = DataProcessor(data_dir=data_dir, input_window=336, output_window=168)
    # Reuse df to save time? No, feature_engineering is same, but create_sequences depends on object state?
    # Actually create_sequences uses self.input_window/output_window.
    # We can reuse df.
    p168.scalers = p24.scalers # Reuse scaler just in case
    p168.feature_columns = p24.feature_columns
    
    X168, _ = p168.create_sequences(df)
    print(f"Output 168h -> Samples: {len(X168)}")

if __name__ == "__main__":
    check_counts()
