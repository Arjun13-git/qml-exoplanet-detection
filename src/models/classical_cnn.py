# src/models/classical_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitCNN(nn.Module):
    def __init__(self, sequence_length=200, num_classes=3):
        super(TransitCNN, self).__init__()
        
        # 1D Convolutional Block 1
        # Input shape: (Batch_Size, Channels, Sequence_Length) -> (B, 1, 200)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Reduces length by half -> 100
        
        # 1D Convolutional Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Reduces length by half -> 50
        
        # Fully Connected Layer Calculation
        # After two MaxPool1d(2) layers, our sequence of 200 becomes 200 / 2 / 2 = 50.
        # We have 32 output channels from conv2. So the flattened size is 32 * 50.
        flattened_size = 32 * 50
        
        # Classifier Head
        self.fc1 = nn.Linear(flattened_size, 64)
        self.dropout = nn.Dropout(0.3) # Prevents overfitting on stellar noise
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape expects: (Batch, Channels, Length)
        # Apply Conv1 + ReLU + Pool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Apply Conv2 + ReLU + Pool
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten for the dense layers
        x = x.view(x.size(0), -1) 
        
        # Dense classification head with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Quick sanity check to ensure the tensor dimensions flow through the network correctly
    print("🧠 Initializing Classical 1D-CNN Baseline...")
    model = TransitCNN(sequence_length=200, num_classes=3)
    
    # Create a dummy batch of 4 preprocessed stars (Batch=4, Channels=1, Length=200)
    dummy_input = torch.randn(4, 1, 200) 
    
    output = model(dummy_input)
    print(f"✅ Forward pass successful!")
    print(f"   ↳ Input shape:  {dummy_input.shape}")
    print(f"   ↳ Output shape: {output.shape} (Expected: 4 items, 3 class logits)")