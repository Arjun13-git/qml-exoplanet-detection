# src/models/train_qnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from hybrid_qnn import HybridQNN

def train_qnn():
    print("🚀 Initializing Quantum Training Pipeline...")
    
    tensor_path = "data/processed/split_tensors.pt"
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"❌ Tensor file missing at {tensor_path}.")
        
    tensors = torch.load(tensor_path)
    X_train = tensors["X_train"].unsqueeze(1)
    y_train = tensors["y_train"]
    X_val = tensors["X_val"].unsqueeze(1)
    y_val = tensors["y_val"]
    
    model = HybridQNN(sequence_length=200, num_classes=3, n_qubits=4, n_layers=2)
    criterion = nn.CrossEntropyLoss()
    # Quantum circuits often benefit from a slightly higher learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    
    epochs = 10
    print(f"⚙️ Commencing {epochs} experimental epochs on mini-batch with QNN...")
    
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
            
    print("✅ Quantum experimental training loop successfully completed!")
    
    os.makedirs("models/saved", exist_ok=True)
    torch.save(model.state_dict(), "models/saved/qnn_experimental_weights.pth")
    print("💾 Experimental weights saved to models/saved/qnn_experimental_weights.pth")

if __name__ == "__main__":
    train_qnn()