# Phase 4.3: Time-Series Transformer (Self-Attention)
**Date:** July 13, 2026
**Pipeline Stage:** Advanced Baseline Implementation

## 1. Primary Objectives
To evaluate a Time-Series Transformer architecture leveraging Multi-Head Self-Attention to mathematically map out the entire 200-point light curve simultaneously, abandoning the sliding-window constraints of CNNs and TCNs.

## 2. Methodology & Execution
* **Architecture (`ts_transformer.py`):** 
    * Implemented strict Positional Encodings using sine/cosine functions to inject temporal order back into the sequence (as self-attention is inherently position-agnostic).
    * Constructed a `TransformerEncoderLayer` with 4 attention heads (`nhead=4`) and an embedding dimension of 64 (`d_model=64`).
* **Training Engine (`train_transformer.py`):** Executed with a reduced learning rate (`lr=0.0005`) to accommodate the mathematical sensitivity of self-attention mechanisms. 

## 3. Outputs & Verification
* **Dimensionality Check:** Passed. The sequence was successfully permuted from PyTorch Conv1d `(Batch, Channels, Length)` standards to NLP-style `(Batch, Length, Channels)`.
* **Gradient Descent:** Passed.

## 4. Conclusion
The Time-Series Transformer baseline is operational and structurally verified.