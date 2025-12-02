import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from src.data_processor import DataProcessor
from src.model import BiLSTM

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10 # Reduced for 1-day build speed
    LEARNING_RATE = 0.001
    INPUT_WINDOW = 336
    OUTPUT_WINDOW = 72
    HIDDEN_SIZE = 128
    LAYERS = 2
    DATA_DIR = "/Users/sk/Desktop/proj"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Preparation
    processor = DataProcessor(data_dir=DATA_DIR, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)
    X_train, y_train, X_test, y_test = processor.get_data_loaders(batch_size=BATCH_SIZE)
    
    # Convert to Tensor
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # Model Initialization
    input_size = X_train.shape[2]
    model = BiLSTM(input_size, HIDDEN_SIZE, OUTPUT_WINDOW, num_layers=LAYERS).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(test_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    # Plot Loss Curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss (Overfitting Check)')
    plt.legend()
    plt.grid(True)
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/loss_curve.png")
    print("Saved artifacts/loss_curve.png")
        
    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bilstm_model.pth")
    print("Model saved to models/bilstm_model.pth")

if __name__ == "__main__":
    train()
