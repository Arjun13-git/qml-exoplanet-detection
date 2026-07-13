# src/models/astronet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Astronet(nn.Module):
    def __init__(self, sequence_length=200, num_classes=3):
        super(Astronet, self).__init__()
        
        # -------------------------------------------------------------
        # GLOBAL BRANCH: Large kernels to capture macro-stellar trends
        # -------------------------------------------------------------
        self.global_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, stride=1, padding=5)
        self.global_pool1 = nn.MaxPool1d(kernel_size=4, stride=4) # Aggressive pooling
        
        self.global_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.global_pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # -------------------------------------------------------------
        # LOCAL BRANCH: Small kernels to capture sharp transit edges
        # -------------------------------------------------------------
        self.local_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.local_pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # Fine-grained pooling
        
        self.local_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.local_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # -------------------------------------------------------------
        # DYNAMIC FLATTENING CALCULATION
        # -------------------------------------------------------------
        # Calculate exactly how many features exit each branch so we can concatenate them
        def _get_conv_output(shape):
            batch_size = 1
            dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
            
            # Run dummy data through Global Branch
            g = self.global_pool1(F.relu(self.global_conv1(dummy_input)))
            g = self.global_pool2(F.relu(self.global_conv2(g)))
            g_size = g.data.view(batch_size, -1).size(1)
            
            # Run dummy data through Local Branch
            l = self.local_pool1(F.relu(self.local_conv1(dummy_input)))
            l = self.local_pool2(F.relu(self.local_conv2(l)))
            l_size = l.data.view(batch_size, -1).size(1)
            
            return g_size, l_size

        g_out_size, l_out_size = _get_conv_output((1, sequence_length))
        total_flattened_size = g_out_size + l_out_size
        
        # -------------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER HEAD
        # -------------------------------------------------------------
        self.fc1 = nn.Linear(total_flattened_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Pass input through Global Branch
        g = self.global_pool1(F.relu(self.global_conv1(x)))
        g = self.global_pool2(F.relu(self.global_conv2(g)))
        g = g.view(g.size(0), -1) # Flatten Global

        # Pass same input through Local Branch
        l = self.local_pool1(F.relu(self.local_conv1(x)))
        l = self.local_pool2(F.relu(self.local_conv2(l)))
        l = l.view(l.size(0), -1) # Flatten Local

        # Concatenate both perspectives
        combined = torch.cat((g, l), dim=1)

        # Pass through dense layers
        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

if __name__ == "__main__":
    print("🔭 Initializing Dual-Branch Astronet Architecture...")
    model = Astronet(sequence_length=200, num_classes=3)
    
    # Dummy experimental batch: (Batch=4, Channels=1, Length=200)
    dummy_input = torch.randn(4, 1, 200) 
    
    output = model(dummy_input)
    print(f"✅ Forward pass successful!")
    print(f"   ↳ Input shape:  {dummy_input.shape}")
    print(f"   ↳ Output shape: {output.shape} (Expected: 4, 3)")