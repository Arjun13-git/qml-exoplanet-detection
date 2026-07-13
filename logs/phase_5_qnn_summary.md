# Phase 5: Hybrid Quantum Neural Network (QNN)
**Date:** July 13, 2026
**Pipeline Stage:** Quantum Architecture Implementation

## 1. Primary Objectives
To engineer a Hybrid Qubit-Based Quantum Neural Network using PennyLane and PyTorch. The architecture must compress 1D astrophysical time-series data into a low-dimensional latent space, encode it into a quantum state, apply strongly entangling unitary operations, and measure the expectation values for ternary classification.

## 2. Methodology & Execution
* **Classical-to-Quantum Bottleneck (`hybrid_qnn.py`):** 
    * Deployed a PyTorch linear layer to compress the 200-point sequence into 4 distinct features.
    * Applied a `tanh` activation scaled by $\pi$ to bound the features between $-\pi$ and $\pi$, preparing them for quantum Angle Embedding.
* **Quantum Circuit (PennyLane):** 
    * Initialized a 4-qubit `default.qubit` simulator.
    * Utilized `AngleEmbedding` for classical data injection and `StronglyEntanglingLayers` to construct highly correlated multi-qubit superpositions.
    * Measured the Pauli-Z observable expectation values on all 4 wires.
* **Hybrid Training Engine (`train_qnn.py`):** Wrapped the PennyLane QNode in a `qml.qnn.TorchLayer` to enable seamless autograd capability across the classical/quantum boundary.

## 3. Outputs & Verification
* **Dimensionality Check:** Passed. The classical data was successfully translated into quantum phase angles and measured back into `(4, 3)` classical logits.
* **Quantum Gradient Descent:** Passed. Backpropagation successfully updated both PyTorch classical weights and PennyLane quantum circuit parameters.

## 4. Conclusion
The Hybrid QNN is fully operational. The experimental mini-batch phase is complete. The architecture is ready to scale to the full MAST dataset.