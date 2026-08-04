import torch
import torch.nn as nn

class AstroNet1D(nn.Module):
    def __init__(self, input_len=200):
        super().__init__()
        # Branch 1: Wide View (Broader stellar context)
        self.wide_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 2: Narrow View (Fine transit details)
        self.narrow_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Combined Classification FC Head
        self.fc = nn.Sequential(
            nn.Linear(64 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        out_wide = self.wide_branch(x).squeeze(-1)
        out_narrow = self.narrow_branch(x).squeeze(-1)
        
        combined = torch.cat([out_wide, out_narrow], dim=1)
        return self.fc(combined)

# Class aliases for dynamic model loading in evaluate scripts
Astronet = AstroNet1D
AstroNet = AstroNet1D

if __name__ == "__main__":
    model = AstroNet1D()
    dummy_input = torch.randn(4, 1, 200)
    out = model(dummy_input)
    print("🔭 AstroNet Dual-Branch Model Initialized Successfully.")
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {out.shape}")