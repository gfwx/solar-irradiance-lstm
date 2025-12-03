import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor
from model import BiLSTM

def forecast_pv():
    # 1. Setup and Load Training Data (to fit scalers)
    data_dir = "/Users/sk/Desktop/proj"
    input_window = 144
    # Output window doesn't matter for fitting scalers
    
    print("Initializing DataProcessor and fitting on training data...")
    processor = DataProcessor(data_dir=data_dir, input_window=input_window, output_window=72)
    
    # Load and process training data to fit scalers and GBDT
    df_train = processor.load_data()
    df_train = processor.feature_engineering(df_train)
    df_train = processor.train_gbdt(df_train)
    df_train = processor.normalize(df_train)
    
    print("Training data processed. Scalers and GBDT fitted.")
    
    # 2. Load Plant Data
    print("\nLoading Plant Data...")
    weather_file = os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv")
    gen_file = os.path.join(data_dir, "Plant_1_Generation_Data.csv")
    
    df_weather = pd.read_csv(weather_file)
    df_gen = pd.read_csv(gen_file)
    
    # 3. Preprocess Plant Data
    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])
    df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
    
    df_weather.set_index('DATE_TIME', inplace=True)
    df_gen.set_index('DATE_TIME', inplace=True)
    
    # Resample to Hourly Mean
    df_weather_hourly = df_weather.resample('h').mean(numeric_only=True)
    df_gen_hourly = df_gen.resample('h').mean(numeric_only=True)
    
    df_plant = df_weather_hourly.rename(columns={
        'AMBIENT_TEMPERATURE': 'temperature',
        'IRRADIATION': 'SWD [W/m**2]'
    })
    
    # Fill missing columns with Training Means
    df_plant['humidity'] = 73.53
    df_plant['pressure'] = 988.82
    df_plant['latitude'] = 8.76
    df_plant['longitude'] = 24.78
    
    target_col = 'SWD [W/m**2]'
    df_plant['GHI_lag_24h'] = df_plant[target_col].shift(24)
    df_plant['GHI_lag_168h'] = df_plant[target_col].shift(168)
    
    df_plant.dropna(inplace=True)
    
    print(f"Plant data shape after preprocessing: {df_plant.shape}")
    
    # 4. Transform Plant Data
    print("Transforming Plant Data...")
    df_plant_scaled = processor.transform_new_data(df_plant)
    
    # Prepare input tensor once
    data_values = df_plant_scaled[processor.feature_columns].values
    X_plant = []
    timestamps = []
    
    for i in range(len(data_values) - input_window):
        X_plant.append(data_values[i : i + input_window])
        timestamps.append(df_plant_scaled.index[i + input_window])
        
    X_plant = np.array(X_plant)
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    X_plant_tensor = torch.tensor(X_plant, dtype=torch.float32).to(device)
    
    horizons = [24, 72, 168]
    results = {}
    
    for horizon in horizons:
        print(f"\nProcessing Horizon: {horizon}h")
        
        # 5. Generate GHI Predictions
        print(f"Loading BiLSTM model for {horizon}h...")
        model = BiLSTM(input_size=len(processor.feature_columns), hidden_size=128, output_size=horizon, num_layers=2).to(device)
        model.load_state_dict(torch.load(f"models/bilstm_{horizon}h.pth", map_location=device))
        model.eval()
        
        print("Generating GHI predictions...")
        batch_size = 32
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_plant_tensor), batch_size):
                batch = X_plant_tensor[i : i + batch_size]
                out = model(batch)
                predictions.append(out.cpu().numpy())
                
        predictions = np.concatenate(predictions, axis=0)
        
        # Use the 1st hour prediction (t+1)
        pred_ghi_1h = predictions[:, 0]
        
        # Inverse transform
        target_idx = processor.feature_columns.index(target_col)
        scaler = processor.scalers['global']
        scale_val = scaler.scale_[target_idx]
        min_val = scaler.min_[target_idx]
        
        pred_ghi_inv = (pred_ghi_1h - min_val) / scale_val
        
        # Clip GHI to 0
        pred_ghi_inv = np.maximum(pred_ghi_inv, 0)
        
        print(f"Horizon {horizon}h - GHI Pred Range: Min {pred_ghi_inv.min():.4f}, Max {pred_ghi_inv.max():.4f}")
        
        pred_timestamps = [ts + pd.Timedelta(hours=1) for ts in timestamps]
        df_pred = pd.DataFrame({'Predicted_GHI': pred_ghi_inv}, index=pred_timestamps)
        
        # 6. Train Optimized Linear Model
        print(f"Training Optimized Linear Model ({horizon}h)...")
        df_reg = df_pred.join(df_gen_hourly['AC_POWER'], how='inner')
        df_reg = df_reg.join(df_plant['temperature'], how='inner')
        
        # Add Time Features
        df_reg['hour'] = df_reg.index.hour
        df_reg['sin_hour'] = np.sin(2 * np.pi * df_reg['hour'] / 24)
        df_reg['cos_hour'] = np.cos(2 * np.pi * df_reg['hour'] / 24)
        
        # Add Polynomial and Interaction Features
        df_reg['GHI_sq'] = df_reg['Predicted_GHI'] ** 2
        df_reg['GHI_cu'] = df_reg['Predicted_GHI'] ** 3
        df_reg['GHI_x_Temp'] = df_reg['Predicted_GHI'] * df_reg['temperature']
        df_reg['GHI_x_cos_hour'] = df_reg['Predicted_GHI'] * df_reg['cos_hour']
        
        df_reg.dropna(inplace=True)
        
        # Features
        features = ['Predicted_GHI', 'temperature', 'GHI_sq', 'GHI_cu', 'GHI_x_Temp', 'sin_hour', 'cos_hour', 'GHI_x_cos_hour']
        X_reg = df_reg[features]
        y_reg = df_reg['AC_POWER']
        
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        
        y_pred_ac = lr.predict(X_reg)
        
        # Clip AC Power to 0
        y_pred_ac = np.maximum(y_pred_ac, 0)
        
        print(f"Horizon {horizon}h - AC Power Pred Range: Min {y_pred_ac.min():.4f}, Max {y_pred_ac.max():.4f}")
        
        r2 = r2_score(y_reg, y_pred_ac)
        rmse = np.sqrt(mean_squared_error(y_reg, y_pred_ac))
        nrmse = rmse / y_reg.mean()
        
        print(f"Results for {horizon}h:")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"nRMSE: {nrmse:.4%}")
        
        results[horizon] = {'R2': r2, 'RMSE': rmse, 'nRMSE': nrmse}
        
        # 7. Plot Sample Forecast
        # Pick a sample to visualize (e.g., middle of the test set)
        sample_idx = 300 # Arbitrary index
        
        # Get History (Actual AC Power)
        # timestamps[sample_idx] is the time of the LAST input step
        # Input window is 144h.
        # We need the AC Power for [timestamps[sample_idx] - 143h ... timestamps[sample_idx]]
        
        t_end_input = timestamps[sample_idx]
        t_start_input = t_end_input - pd.Timedelta(hours=input_window - 1)
        
        history_slice = df_gen_hourly.loc[t_start_input : t_end_input]['AC_POWER']
        
        # Get Forecast
        # We need to generate the specific forecast for this sample
        # The 'predictions' array contains flattened 1-step-ahead predictions for the whole series.
        # We need the multi-step forecast for THIS sample.
        
        # Re-run model for this single sample to get the full horizon prediction
        sample_input = X_plant_tensor[sample_idx].unsqueeze(0) # (1, 144, Features)
        with torch.no_grad():
            sample_pred_ghi_seq = model(sample_input).cpu().numpy()[0] # (Horizon,)
            
        # Inverse transform GHI
        sample_pred_ghi_inv = (sample_pred_ghi_seq - min_val) / scale_val
        sample_pred_ghi_inv = np.maximum(sample_pred_ghi_inv, 0)
        
        # Get Temperature for the forecast period
        # t_start_forecast = t_end_input + 1h
        t_start_forecast = t_end_input + pd.Timedelta(hours=1)
        t_end_forecast = t_end_input + pd.Timedelta(hours=horizon)
        
        # We need actual temperature for these timestamps
        # Note: In a real scenario we'd use forecasted temp, here we use actuals
        future_temps = df_plant.loc[t_start_forecast : t_end_forecast]['temperature']
        
        # If we run out of data (at the end of series), clip
        if len(future_temps) < horizon:
            print(f"Warning: Not enough future data for sample {sample_idx}. Skipping plot.")
            continue
            
        # Prepare regression input
        df_sample_reg = pd.DataFrame({
            'Predicted_GHI': sample_pred_ghi_inv,
            'temperature': future_temps.values
        }, index=future_temps.index)
        
        # Add Time Features
        df_sample_reg['hour'] = df_sample_reg.index.hour
        df_sample_reg['sin_hour'] = np.sin(2 * np.pi * df_sample_reg['hour'] / 24)
        df_sample_reg['cos_hour'] = np.cos(2 * np.pi * df_sample_reg['hour'] / 24)
        
        # Add Polynomial and Interaction Features
        df_sample_reg['GHI_sq'] = df_sample_reg['Predicted_GHI'] ** 2
        df_sample_reg['GHI_cu'] = df_sample_reg['Predicted_GHI'] ** 3
        df_sample_reg['GHI_x_Temp'] = df_sample_reg['Predicted_GHI'] * df_sample_reg['temperature']
        df_sample_reg['GHI_x_cos_hour'] = df_sample_reg['Predicted_GHI'] * df_sample_reg['cos_hour']
        
        # Predict AC Power
        features = ['Predicted_GHI', 'temperature', 'GHI_sq', 'GHI_cu', 'GHI_x_Temp', 'sin_hour', 'cos_hour', 'GHI_x_cos_hour']
        sample_pred_ac = lr.predict(df_sample_reg[features])
        sample_pred_ac = np.maximum(sample_pred_ac, 0)
        
        df_sample_ac = pd.DataFrame({'AC_POWER': sample_pred_ac}, index=future_temps.index)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot History
        plt.plot(history_slice.index, history_slice.values, label='History (Actual)', color='blue')
        
        # Plot Forecast
        plt.plot(df_sample_ac.index, df_sample_ac['AC_POWER'], label=f'Forecast ({horizon}h)', color='orange', linestyle='--')
        
        # Optional: Plot Actual Future for comparison (Dashed Grey)
        actual_future = df_gen_hourly.loc[t_start_forecast : t_end_forecast]['AC_POWER']
        plt.plot(actual_future.index, actual_future.values, label='Actual Future', color='grey', alpha=0.5, linestyle=':')
        
        plt.title(f'PV Power Forecast Sample ({horizon}h Horizon)\nModel R2: {r2:.2f}')
        plt.xlabel('Time')
        plt.ylabel('AC Power')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'artifacts/pv_power_forecast_{horizon}h.png')
        print(f"Plot saved to artifacts/pv_power_forecast_{horizon}h.png")
        plt.close()

if __name__ == "__main__":
    forecast_pv()

if __name__ == "__main__":
    forecast_pv()
