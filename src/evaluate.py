import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_processor import DataProcessor
from src.model import BiLSTM

def evaluate():
    # Hyperparameters (must match train.py)
    BATCH_SIZE = 32
    INPUT_WINDOW = 336
    OUTPUT_WINDOW = 72
    HIDDEN_SIZE = 128
    LAYERS = 2
    DATA_DIR = "/Users/sk/Desktop/proj"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    processor = DataProcessor(data_dir=DATA_DIR, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)
    _, _, X_test, y_test = processor.get_data_loaders(batch_size=BATCH_SIZE)
    
    # Load Model
    input_size = X_test.shape[2]
    model = BiLSTM(input_size, HIDDEN_SIZE, OUTPUT_WINDOW, num_layers=LAYERS).to(device)
    model.load_state_dict(torch.load("models/bilstm_model.pth", map_location=device))
    model.eval()
    
    # Predict
    print("Generating predictions...")
    X_tensor = torch.Tensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
        
    # Inverse Transform
    # We need to inverse transform only the target column.
    # The scaler was fitted on all feature columns.
    # We can reconstruct a dummy array to inverse transform.
    
    scaler = processor.scalers['global']
    target_idx = processor.feature_columns.index(processor.target_column)
    
    def inverse_transform_target(y_scaled):
        # y_scaled shape: (samples, output_window)
        # We need to apply inverse transform for each timestep
        # Construct a dummy matrix of shape (samples * output_window, n_features)
        
        n_features = len(processor.feature_columns)
        y_flat = y_scaled.flatten()
        
        dummy = np.zeros((len(y_flat), n_features))
        dummy[:, target_idx] = y_flat
        
        # Inverse transform
        dummy_inv = scaler.inverse_transform(dummy)
        
        # Extract target
        y_inv = dummy_inv[:, target_idx]
        return y_inv.reshape(y_scaled.shape)

    y_test_inv = inverse_transform_target(y_test)
    y_pred_inv = inverse_transform_target(y_pred)
    
    # Metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    # Normalized RMSE (using mean of true data)
    nrmse = rmse / np.mean(y_test_inv)
    
    # Skill Score vs Persistence
    # Persistence: Predict the last 3 days (72h) for the next 3 days
    # Input window is 336h. Last 72h of input.
    
    # Extract target feature from X_test (last 72 timesteps)
    X_test_target_scaled = X_test[:, -72:, target_idx]
    y_persistence_inv = inverse_transform_target(X_test_target_scaled)
    
    rmse_persistence = np.sqrt(mean_squared_error(y_test_inv, y_persistence_inv))
    skill_score = 1 - (rmse / rmse_persistence)
    
    print(f"\nEvaluation Results:")
    print(f"RMSE: {rmse:.2f} W/m^2")
    print(f"nRMSE: {nrmse:.2%}")
    print(f"MAE: {mae:.2f} W/m^2")
    print(f"Skill Score (vs 3-Day Persistence): {skill_score:.2f}")
    
    # Plotting
    os.makedirs("artifacts", exist_ok=True)
    
    # Plot 1: Sample Forecasts
    plt.figure(figsize=(15, 10))
    for i in range(min(3, len(y_test_inv))):
        plt.subplot(3, 1, i+1)
        plt.plot(y_test_inv[i], label='Actual', color='black')
        plt.plot(y_pred_inv[i], label='Forecast (Hybrid)', color='blue')
        plt.plot(y_persistence_inv[i], label='Persistence', color='gray', linestyle='--')
        plt.title(f"Sample {i+1}: 3-Day Forecast")
        plt.ylabel("Irradiance (W/m^2)")
        plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/forecast_samples.png")
    print("Saved forecast_samples.png")
    
    # Plot 2: Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_inv.flatten(), y_pred_inv.flatten(), alpha=0.1, s=1)
    plt.plot([0, 1200], [0, 1200], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Irradiance")
    plt.savefig("artifacts/scatter_plot.png")
    print("Saved scatter_plot.png")

if __name__ == "__main__":
    evaluate()
