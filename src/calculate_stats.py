import sys
import os

# Add the current directory to sys.path to import data_processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor

def calculate_stats():
    # Initialize processor
    # Assuming the script is running from src, data_dir should be up one level
    # But the user's workspace is /Users/sk/Desktop/proj, so data_dir is that.
    data_dir = "/Users/sk/Desktop/proj"
    processor = DataProcessor(data_dir=data_dir)
    
    print("Loading data...")
    df = processor.load_data()
    total_hours = len(df)
    print(f"Total hours of data (rows in combined dataframe): {total_hours}")
    
    print("Running full pipeline to get exact sample count...")
    # We need to run the full pipeline because feature engineering drops rows (lags)
    # and create_sequences logic depends on available data per site.
    
    # feature_engineering
    df = processor.feature_engineering(df)
    
    # train_gbdt (might take a moment but needed for consistency if it drops anything, though it shouldn't drop rows, just adds column)
    # Actually train_gbdt splits data, trains, and predicts on ALL X. It doesn't drop rows.
    # But let's run it to be safe or mock it if it's too slow. 
    # It uses LightGBM, should be fast enough for this size.
    df = processor.train_gbdt(df)
    
    # normalize
    df = processor.normalize(df)
    
    # create_sequences
    X, y = processor.create_sequences(df)
    
    print(f"Total samples generated: {len(X)}")

if __name__ == "__main__":
    calculate_stats()
