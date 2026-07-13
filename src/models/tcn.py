# src/models/tcn.py
import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    """A single TCN block with dilated convolutions and residual connections."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Calculate padding needed to maintain sequence length before causal trimming
        self.padding = (kernel_size - 1) * dilation
        
        # First Dilated Convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=self.padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second Dilated Convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=self.padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection: downsample if channel dimensions change
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # Pass 1 with Causal Trimming (removing the padded "future" data)
        out = self.conv1(x)
        out = out[:, :, :-self.padding] 
        out = self.dropout1(self.relu1(out))
        
        # Pass 2 with Causal Trimming
        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.dropout2(self.relu2(out))
        
        # Add residual connection to prevent vanishing gradients
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

class TransitTCN(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 32, 64], kernel_size=3, dropout=0.3, num_classes=3):
        super(TransitTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Build the network with exponentially increasing dilations (1, 2, 4...)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation_size, dropout))
            
        self.network = nn.Sequential(*layers)
        
        # TCN outputs a sequence. We use a fully connected layer on the final time step.
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        out = self.network(x)
        # Extract the very last feature vector which has "seen" the entire sequence
        out = out[:, :, -1] 
        return self.fc(out)

if __name__ == "__main__":
    print("🕰️ Initializing Temporal Convolutional Network (TCN)...")
    model = TransitTCN()
    
    # Dummy experimental batch: (Batch=4, Channels=1, Length=200)
    dummy_input = torch.randn(4, 1, 200) 
    
    output = model(dummy_input)
    print(f"✅ Forward pass successful!")
    print(f"   ↳ Input shape:  {dummy_input.shape}")
    print(f"   ↳ Output shape: {output.shape} (Expected: 4, 3)")