import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_dir, input_window=336, output_window=72):
        self.data_dir = data_dir
        self.input_window = input_window
        self.output_window = output_window
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'SWD [W/m**2]' # Short-wave downward (Global Horizontal Irradiance)

    def load_data(self):
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*_radiation_*.tab")))
        print(f"Found {len(all_files)} files.")
        
        dfs = []
        for file in all_files:
            print(f"Processing {file}...")
            try:
                # BSRN .tab files have metadata at the top, usually ending with */
                # We need to find the line number where the data starts
                with open(file, 'r', encoding='latin1') as f:
                    lines = f.readlines()
                
                header_row = 0
                lat, lon = 0.0, 0.0
                for i, line in enumerate(lines):
                    if "Coverage:" in line and "LATITUDE:" in line:
                        try:
                            parts = line.split('*')
                            for part in parts:
                                if "LATITUDE:" in part:
                                    lat = float(part.split("LATITUDE:")[1].strip())
                                if "LONGITUDE:" in part:
                                    lon = float(part.split("LONGITUDE:")[1].strip())
                        except:
                            print(f"Warning: Could not parse lat/lon from {line}")
                            
                    if line.strip().startswith("Date/Time"):
                        header_row = i
                        break
                
                print(f"Detected header at row {header_row}, Lat: {lat}, Lon: {lon}")
                
                # Read data
                df = pd.read_csv(file, sep='\t', skiprows=header_row, encoding='latin1')
                
                # Extract site code from filename (e.g., ABS_radiation...)
                site_code = os.path.basename(file).split('_')[0]
                df['site'] = site_code
                df['latitude'] = lat
                df['longitude'] = lon
                
                # Parse Date/Time
                # Format example: 2025-05-01T00:00
                df['Date/Time'] = pd.to_datetime(df['Date/Time'])
                df.set_index('Date/Time', inplace=True)
                
                # Clean data
                # Replace negative radiation values with 0
                rad_cols = [c for c in df.columns if '[W/m**2]' in c]
                for col in rad_cols:
                    df[col] = df[col].apply(lambda x: max(0, x))
                
                # Handle missing values (simple forward fill then backward fill for now)
                df.replace(r'^\s*$', np.nan, regex=True, inplace=True) # Replace empty strings
                df = df.apply(pd.to_numeric, errors='coerce') # Convert all to numeric
                
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                
                # Resample to hourly
                # We need to preserve lat/lon/site which are constant
                # Group by site (and lat/lon) and resample
                df_hourly = df.resample('H').mean()
                df_hourly['site'] = site_code
                df_hourly['latitude'] = lat
                df_hourly['longitude'] = lon
                
                dfs.append(df_hourly)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
        if not dfs:
            raise ValueError("No data loaded!")
            
        combined_df = pd.concat(dfs)
        return combined_df

    def feature_engineering(self, df):
        print("Starting feature engineering...")
        
        # 1. Time features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 2. Lag features (Global Radiation)
        # We need to be careful with lags across different sites. 
        # Group by site to calculate lags correctly.
        df['GHI_lag_24h'] = df.groupby('site')[self.target_column].shift(24)
        df['GHI_lag_168h'] = df.groupby('site')[self.target_column].shift(168)
        
        # 3. Meteorological features (if available)
        # Robust mapping using substring matching to handle encoding issues with special chars
        col_map = {}
        for col in df.columns:
            if "T2" in col and "C" in col: # T2 [°C] or T2 [Â°C]
                col_map['temperature'] = col
            elif "RH" in col and "%" in col:
                col_map['humidity'] = col
            elif "PoPoPoPo" in col or ("pressure" in col.lower() and "hpa" in col.lower()):
                col_map['pressure'] = col
        
        for new_name, original_col in col_map.items():
            df[new_name] = df[original_col]
            
        # Ensure all expected columns exist, fill with 0 if not found
        for expected in ['temperature', 'humidity', 'pressure']:
            if expected not in df.columns:
                df[expected] = 0.0
                
        # Drop rows with NaNs created by shifting
        df.dropna(inplace=True)
        
        # Define feature columns
        self.feature_columns = [
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
            'GHI_lag_24h', 'GHI_lag_168h',
            'temperature', 'humidity', 'pressure',
            'latitude', 'longitude',
            self.target_column # Include target in features for autoregression context if needed
        ]
        
        return df

    def train_gbdt(self, df):
        print("Training GBDT for feature extraction...")
        # Features for GBDT (exclude target and future info)
        # We use the features we've already engineered
        X = df[self.feature_columns].drop(columns=[self.target_column], errors='ignore')
        y = df[self.target_column]
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data], 
                          callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]) # Silence log
        
        # Predict and add as feature
        print("Adding GBDT predictions as feature...")
        df['gbdt_pred'] = model.predict(X)
        
        # Add to feature columns
        self.feature_columns.append('gbdt_pred')
        
        return df

    def normalize(self, df):
        print("Normalizing data...")
        # Normalize features per site or globally? Globally is simpler for now given the task constraints.
        # But per-site normalization for radiation might be better if climates vary wildly.
        # Let's do global normalization for simplicity and robustness.
        
        scaler = MinMaxScaler()
        df[self.feature_columns] = scaler.fit_transform(df[self.feature_columns])
        self.scalers['global'] = scaler
        return df

    def create_sequences(self, df):
        print("Creating sequences...")
        X, y = [], []
        
        # Process each site separately to avoid boundary issues
        for site, site_df in df.groupby('site'):
            data = site_df[self.feature_columns].values
            target = site_df[self.target_column].values # Target is GHI
            
            # Sliding window
            # We need input_window + output_window size
            total_window = self.input_window + self.output_window
            
            if len(data) < total_window:
                continue
                
            for i in range(len(data) - total_window):
                X.append(data[i : i + self.input_window])
                y.append(target[i + self.input_window : i + total_window])
                
        return np.array(X), np.array(y)

    def get_data_loaders(self, batch_size=32):
        df = self.load_data()
        df = self.feature_engineering(df)
        df = self.train_gbdt(df) # Add GBDT feature
        df = self.normalize(df)
        X, y = self.create_sequences(df)
        
        print(f"Generated sequences: X shape {X.shape}, y shape {y.shape}")
        
        # Split into train/test (time-based split not strictly possible with mixed sites easily without leakage if we just shuffle, 
        # but for this task we'll just split by index assuming data is roughly ordered or just take last chunk)
        # Better: Split by site or just take last 20% of sequences.
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test the processor
    processor = DataProcessor(data_dir="/Users/sk/Desktop/proj")
    X_train, y_train, X_test, y_test = processor.get_data_loaders()
    print("Data processing test complete.")
