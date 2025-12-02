import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) 
        
        # Decode the hidden state of the last time step
        # out shape: (batch_size, seq_len, hidden_size * 2)
        # We want to use the last time step's output to predict the next sequence?
        # Or are we doing seq2seq?
        # The prompt implies: input window -> predict next week (168 hours).
        # We can take the last hidden state and project it to 168 outputs.
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# class GBDTWrapper:
#     def __init__(self):
#         self.model = None
        
#     def fit(self, X_train, y_train):
#         # X_train shape: (samples, timesteps, features)
#         # y_train shape: (samples, output_steps)
#         # Flatten for GBDT: We need to predict the *next* step or the whole sequence?
#         # Standard GBDT doesn't handle sequences well natively.
#         # Strategy: Train GBDT to predict the *mean* of the next week, or train 168 separate models?
#         # "Simpler to implement... Handles structural patterns".
#         # Let's try to predict the next hour's value based on current features, 
#         # effectively acting as a sophisticated feature extractor for "what GBDT thinks the radiation should be".
#         # We will reshape input to (samples * timesteps, features) and target to (samples * timesteps, 1) 
#         # but we don't have targets for every timestep in the input window if we only have y for the output window.
        
#         # Alternative Hybrid Strategy:
#         # Train GBDT on the flattened input window to predict the *first* hour of the output, 
#         # or use it to predict residuals.
        
#         # Let's stick to the prompt's "Stage 1: GBDT for feature extraction".
#         # We'll train it to predict the target variable using the input features in a tabular way.
#         # We can treat every timestep in our historical data as a training point.
#         # X_train_flat = X_train.reshape(-1, X_train.shape[-1])
#         # But we need targets aligned with inputs.
#         # In `create_sequences`, we have inputs X and targets y (future).
#         # We can't easily train GBDT on X->y directly if y is a sequence.
        
#         # Simplified robust approach:
#         # We won't train GBDT *inside* the sequence generation loop.
#         # We will assume the GBDT is trained on the raw dataframe (row by row) to predict 'current' radiation 
#         # given 'current' features (like zenith, time). This captures the "structural patterns" (e.g. clear sky model).
#         # Then we pass this "structural prediction" as a feature to the LSTM.
#         pass

#     def predict(self, X):
#         # Placeholder
#         return np.zeros((X.shape[0], X.shape[1], 1))
