# src/models/train_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from ts_transformer import TransitTransformer

def train_transformer():
    print("🚀 Initializing Transformer Training Pipeline...")
    
    tensor_path = "data/processed/split_tensors.pt"
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"❌ Tensor file missing at {tensor_path}.")
        
    tensors = torch.load(tensor_path)
    X_train = tensors["X_train"].unsqueeze(1)
    y_train = tensors["y_train"]
    X_val = tensors["X_val"].unsqueeze(1)
    y_val = tensors["y_val"]
    
    # Initialize the Transformer
    model = TransitTransformer(sequence_length=200, num_classes=3, d_model=64, nhead=4, num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower learning rate often favors Transformers
    
    epochs = 10
    print(f"⚙️ Commencing {epochs} experimental epochs on mini-batch with Transformer...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            _, predicted = torch.max(val_outputs, 1)
            correct = (predicted == y_val).sum().item()
            val_accuracy = correct / len(y_val) * 100
            
        if (epoch + 1) % 2 == 0:
            print(f"   ↳ Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_accuracy:.1f}%")
            
    print("✅ Transformer experimental training loop successfully completed!")
    
    os.makedirs("models/saved", exist_ok=True)
    torch.save(model.state_dict(), "models/saved/transformer_experimental_weights.pth")
    print("💾 Experimental weights saved to models/saved/transformer_experimental_weights.pth")

if __name__ == "__main__":
    train_transformer()