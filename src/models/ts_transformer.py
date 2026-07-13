# src/models/ts_transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects order information into the sequence embeddings."""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, d_model)
        return x + self.pe[:, :x.size(1)]

class TransitTransformer(nn.Module):
    def __init__(self, sequence_length=200, num_classes=3, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransitTransformer, self).__init__()
        
        # 1. Project the 1D input flux (1 feature per step) to d_model embedding space
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        # 2. Build the Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classifier head mapping the pooled encoder output to classes
        self.fc1 = nn.Linear(d_model * sequence_length, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (Batch, Channels=1, Length=200)
        # Permute to match text/sequence expectations: (Batch, Length, Channels=1)
        x = x.permute(0, 2, 1)
        
        # Project and apply positional encodings
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Pass through the self-attention blocks
        x = self.transformer_encoder(x)
        
        # Flatten the sequence dimensions for classification
        x = x.contiguous().view(x.size(0), -1)
        
        # Head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    print("🤖 Initializing Time-Series Self-Attention Transformer...")
    model = TransitTransformer()
    
    # Dummy experimental batch: (Batch=4, Channels=1, Length=200)
    dummy_input = torch.randn(4, 1, 200) 
    
    output = model(dummy_input)
    print(f"✅ Forward pass successful!")
    print(f"   ↳ Input shape:  {dummy_input.shape}")
    print(f"   ↳ Output shape: {output.shape} (Expected: 4, 3)")