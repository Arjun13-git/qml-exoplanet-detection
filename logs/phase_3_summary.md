# Phase 3: Classical Baseline Architecture (1D-CNN)
**Date:** July 13, 2026
**Pipeline Stage:** Model Construction and Experimental Training Loop
**Status:** ⚠️ EXPERIMENTAL PHASE - Validated on Mini-Batch

## 1. Primary Objectives
To construct a classical 1D Convolutional Neural Network (CNN) to serve as the baseline performance metric for future Quantum Machine Learning (QML) comparisons. The architecture must successfully ingest 1D time-series transit arrays and output ternary classification logits.

## 2. Methodology & Execution
* **Model Architecture (`classical_cnn.py`):** 
    * Designed a PyTorch-based neural network featuring two `Conv1d` layers for spatial feature extraction (detecting the sharp edges of planetary transits).
    * Integrated `MaxPool1d` layers for progressive downsampling of the time-series array.
    * Added a fully connected classification head with Dropout (0.3) to mitigate overfitting on stellar noise.
* **Training Engine (`train_classical.py`):** 
    * Built the backpropagation loop utilizing the Adam optimizer (`lr=0.001`) and Cross-Entropy Loss.
    * Engineered the tensor loading mechanism to inject the `(Batch, Channels, Length)` dimensional requirements of PyTorch's 1D convolutional layers.
    * Executed a 10-epoch validation run.

## 3. Outputs & Verification
* **Dimensionality Check:** Passed. The network successfully digests `(4, 1, 200)` input tensors and outputs `(4, 3)` classification probabilities.
* **Gradient Descent:** Passed. Training loss successfully descended across the experimental epochs, proving that backpropagation and weight updates are functioning.
* **Model Weights:** Saved untreated experimental weights to `models/saved/cnn_experimental_weights.pth`.

## 4. Conclusion & Next Steps
The classical baseline is fully operational and mathematically sound. 
**Pending Action:** Transition to Phase 4 to design the QML architecture (Dimensionality Reduction via Autoencoders and Variational Quantum Circuits) to challenge this classical baseline.