# src/models/hybrid_qnn.py
import torch
import torch.nn as nn
import pennylane as qml

# 1. Define the Quantum Device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 2. Design the Parameterized Quantum Circuit (PQC)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # inputs shape: (n_qubits,)
    # weights shape: (n_layers, n_qubits, 3)
    
    # Angle Embedding: Encodes classical data into the rotation angles of the qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Strongly Entangling Layers: Creates complex quantum correlations (superposition & entanglement)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure the expectation value of the Pauli-Z observable on each qubit
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class HybridQNN(nn.Module):
    def __init__(self, sequence_length=200, num_classes=3, n_qubits=4, n_layers=2):
        super(HybridQNN, self).__init__()
        self.n_qubits = n_qubits
        
        # --- CLASSICAL DIMENSIONALITY REDUCTION ---
        # Squeezes the 200-point light curve down to just 4 distinct features
        self.fc_in = nn.Linear(sequence_length, n_qubits)
        
        # --- THE QUANTUM LAYER ---
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # --- CLASSICAL CLASSIFICATION HEAD ---
        # Takes the quantum expectation values and maps them to the 3 classes
        self.fc_out = nn.Linear(n_qubits, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Channels=1, Length=200) -> Flatten
        x = x.view(x.size(0), -1) 
        
        # Classical Reduction
        x = self.fc_in(x)
        
        # Bound the values between -π and π for quantum phase encoding
        x = torch.tanh(x) * 3.14159 
        
        # Enter the Quantum Realm
        x = self.qlayer(x)
        
        # Map quantum measurements to class probabilities
        x = self.fc_out(x)
        return x

if __name__ == "__main__":
    print("🌌 Initializing Hybrid Quantum Neural Network (QNN)...")
    model = HybridQNN()
    
    # Dummy experimental batch
    dummy_input = torch.randn(4, 1, 200) 
    
    output = model(dummy_input)
    print(f"✅ Quantum forward pass successful!")
    print(f"   ↳ Input shape:  {dummy_input.shape}")
    print(f"   ↳ Output shape: {output.shape} (Expected: 4, 3)")