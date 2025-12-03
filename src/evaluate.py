import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_processor import DataProcessor
from src.model import BiLSTM

import argparse

def evaluate(output_window=72, model_path="models/bilstm_model.pth"):
    # Hyperparameters (must match train.py)
    BATCH_SIZE = 32
    INPUT_WINDOW = 144
    OUTPUT_WINDOW = output_window
    HIDDEN_SIZE = 128
    LAYERS = 2
    DATA_DIR = "/Users/sk/Desktop/proj"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load Data
    processor = DataProcessor(data_dir=DATA_DIR, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)
    _, _, X_test, y_test = processor.get_data_loaders(batch_size=BATCH_SIZE)
    
    # Load Model
    input_size = X_test.shape[2]
    model = BiLSTM(input_size, HIDDEN_SIZE, OUTPUT_WINDOW, num_layers=LAYERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Predict
    print(f"Generating predictions for {OUTPUT_WINDOW}h horizon...")
    X_tensor = torch.Tensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()
        
    # Inverse Transform
    scaler = processor.scalers['global']
    target_idx = processor.feature_columns.index(processor.target_column)
    
    def inverse_transform_target(y_scaled):
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
    
    # R2 Score (Coefficient of Determination)
    # R2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_test_inv - y_pred_inv) ** 2)
    ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Skill Score vs Persistence
    # Persistence: Predict the last output_window hours for the next output_window hours?
    # Or just last 24h repeated?
    # Simple persistence: Last observed value repeated?
    # Given the prompt "Predict the last 3 days (72h) for the next 3 days" in original code.
    # We will adapt persistence to use the last `output_window` of input as the prediction.
    
    # Persistence: Predict the last value repeated for the horizon
    # This avoids shape mismatch if Output Window > Input Window
    last_values = X_test[:, -1, target_idx] # Shape: (Batch,)
    y_persistence_scaled = np.tile(last_values.reshape(-1, 1), (1, OUTPUT_WINDOW)) # Shape: (Batch, Output)
    y_persistence_inv = inverse_transform_target(y_persistence_scaled)
    
    rmse_persistence = np.sqrt(mean_squared_error(y_test_inv, y_persistence_inv))
    skill_score = 1 - (rmse / rmse_persistence)
    
    print(f"\nEvaluation Results ({OUTPUT_WINDOW}h):")
    print(f"RMSE: {rmse:.2f} W/m^2")
    print(f"nRMSE: {nrmse:.2%}")
    print(f"MAE: {mae:.2f} W/m^2")
    print(f"R2 Score: {r2:.4f}")
    print(f"Skill Score: {skill_score:.2f}")
    
    # Plotting
    os.makedirs("artifacts", exist_ok=True)
    
    # Plot first 100 hours of the first sequence in test set (or just first sequence)
    # y_test_inv shape: (samples, output_window)
    # Let's plot the first sequence prediction vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv[0], label='Actual')
    plt.plot(y_pred_inv[0], label='Predicted')
    plt.title(f'Actual vs Predicted GHI ({OUTPUT_WINDOW}h Horizon) - Sample 0')
    plt.xlabel('Time (Hours)')
    plt.ylabel('GHI (W/m^2)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"artifacts/pred_vs_actual_{OUTPUT_WINDOW}h.png")
    plt.close()
    
    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_inv.flatten(), y_pred_inv.flatten(), alpha=0.1, s=1)
    plt.plot([0, 1200], [0, 1200], 'r--') # Diagonal
    plt.title(f'Scatter Plot ({OUTPUT_WINDOW}h Horizon)')
    plt.xlabel('Actual GHI')
    plt.ylabel('Predicted GHI')
    plt.xlim(0, 1200)
    plt.ylim(0, 1200)
    plt.grid(True)
    plt.savefig(f"artifacts/scatter_{OUTPUT_WINDOW}h.png")
    plt.close()

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "r2": r2,
        "skill_score": skill_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_window", type=int, default=72)
    parser.add_argument("--model_path", type=str, default="models/bilstm_model.pth")
    args = parser.parse_args()
    
    evaluate(output_window=args.output_window, model_path=args.model_path)
