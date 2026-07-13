# src/models/train_astronet.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from astronet import Astronet

def train_astronet():
    print("🚀 Initializing Astronet Training Pipeline...")
    
    # 1. Load the preprocessed experimental tensors
    tensor_path = "data/processed/split_tensors.pt"
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"❌ Tensor file missing at {tensor_path}. Run Phase 2 first.")
        
    tensors = torch.load(tensor_path)
    
    # Conv1d expects input shape: (Batch, Channels, Length)
    X_train = tensors["X_train"].unsqueeze(1)
    y_train = tensors["y_train"]
    X_val = tensors["X_val"].unsqueeze(1)
    y_val = tensors["y_val"]
    
    print(f"📦 Loaded Training Data Shape: {X_train.shape}")
    
    # 2. Instantiate the Astronet model, loss function, and optimizer
    model = Astronet(sequence_length=200, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Mini-Batch Training Loop
    epochs = 10
    print(f"⚙️ Commencing {epochs} experimental epochs on mini-batch with Astronet...")
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through dual branches
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass (calculate gradients for both branches)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Validation pass
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            # Get predictions for accuracy tracking
            _, predicted = torch.max(val_outputs, 1)
            correct = (predicted == y_val).sum().item()
            val_accuracy = correct / len(y_val) * 100
            
        if (epoch + 1) % 2 == 0:
            print(f"   ↳ Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_accuracy:.1f}%")
            
    print("✅ Astronet experimental training loop successfully completed!")
    
    # Save the experimental weights
    os.makedirs("models/saved", exist_ok=True)
    torch.save(model.state_dict(), "models/saved/astronet_experimental_weights.pth")
    print("💾 Experimental weights saved to models/saved/astronet_experimental_weights.pth")

if __name__ == "__main__":
    train_astronet()