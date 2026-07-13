# src/models/ssm_mamba.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredStateSpaceBlock(nn.Module):
    """
    A pure-PyTorch simplified Structured State Space (SSM) block.
    Models the continuous-time differential equation: h'(t) = Ah(t) + Bx(t)
    Discretized for time-series data.
    """
    def __init__(self, d_model, d_state=16):
        super(StructuredStateSpaceBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters (A, B, C) and step size (dt)
        # These are the matrices that govern the differential equations
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.1)
        
        self.linear_in = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: (Batch, Sequence_Length, d_model)
        Batch, Length, D = x.shape
        x_proj = self.linear_in(x)
        
        # Discretize continuous time parameters
        dt = torch.exp(self.log_dt) # (D,)
        # FORCE A TO BE STRICTLY NEGATIVE to prevent exponential explosion
        A_bar = torch.exp(-torch.abs(self.A) * dt.unsqueeze(-1)) # (D, d_state)
        B_bar = self.B * dt.unsqueeze(-1) # (D, d_state)
        
        # Initialize hidden state
        h = torch.zeros(Batch, D, self.d_state, device=x.device)
        outputs = []
        
        # Sequential scan (linear time recurrence)
        for t in range(Length):
            u_t = x_proj[:, t, :].unsqueeze(-1) # (Batch, D, 1)
            h = h * A_bar + B_bar * u_t # Update latent state
            y_t = torch.sum(h * self.C, dim=-1) # Project to output
            outputs.append(y_t.unsqueeze(1))
            
        y = torch.cat(outputs, dim=1)
        return self.linear_out(y)

class TransitSSM(nn.Module):
    def __init__(self, sequence_length=200, num_classes=3, d_model=32, d_state=16, num_layers=2):
        super(TransitSSM, self).__init__()
        
        # Feature projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Stacked State Space Blocks
        self.ssm_layers = nn.ModuleList([
            StructuredStateSpaceBlock(d_model, d_state) for _ in range(num_layers)
        ])
        
        # Classification Head
        self.fc1 = nn.Linear(d_model * sequence_length, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Channels=1, Length=200) -> Permute to (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        
        for layer in self.ssm_layers:
            x = F.relu(layer(x))
            
        # Flatten and classify
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    print("🧬 Initializing Structured State Space Model (SSM)...")
    model = TransitSSM()
    
    dummy_input = torch.randn(4, 1, 200) 
    output = model(dummy_input)
    
    print(f"✅ Forward pass successful!")
    print(f"   ↳ Input shape:  {dummy_input.shape}")
    print(f"   ↳ Output shape: {output.shape} (Expected: 4, 3)")