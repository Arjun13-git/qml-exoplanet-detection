# Phase 4.2: Temporal Convolutional Network (TCN)
**Date:** July 13, 2026
**Pipeline Stage:** Advanced Baseline Implementation

## 1. Primary Objectives
To implement a Temporal Convolutional Network (TCN) featuring dilated causal convolutions to model long-range temporal dependencies across the 200-point transit sequence without losing fine-grained edge resolution.

## 2. Methodology & Execution
* **Architecture (`tcn.py`):** 
    * Built a network with exponentially expanding dilations ($2^0, 2^1, 2^2$) across 3 levels to broaden the receptive field dynamically.
    * Enforced causal trimming to prevent data leakage from future time steps into past steps.
    * Integrated residual connections to ensure stable gradient flows back through the deep layers.
* **Training Engine (`train_tcn.py`):** Verified compilation and backpropagation over 10 experimental epochs.

## 3. Outputs & Verification
* **Dimensionality Check:** Passed. Outflow shaped to `(4, 3)` ternary logits.
* **Gradient Descent:** Passed. Train Loss dropped from 1.0630 to 0.9039.

## 4. Conclusion
The TCN architecture is verified and fully operational.