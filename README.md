# Credit Risk Prediction using Graph Neural Networks with Feature Attention

**A scalable, interpretable, and high-performance GNN-based pipeline for credit default risk prediction, integrating feature attention, dynamic graph construction, and sequential modeling.**
---

## Overview

This project presents **FA-GNN (Feature Attention Graph Neural Network)** — a novel approach to credit risk prediction that:

* Models borrowers as graph nodes with dynamic edges based on feature similarity and time patterns.
* Utilizes **multi-head feature attention** to focus on salient financial information.
* Integrates **LSTM layers** to model temporal dependencies.
* Enhances interpretability using **SHAP** and **t-SNE** visualizations.
* Achieves **state-of-the-art performance (AUC: 0.77067)** on the **Home Credit Default Risk** dataset.

---

## Key Features

* **Graph Neural Networks**: Dynamically constructed borrower graphs based on feature and temporal similarity.
* **Feature Attention**: Multi-head attention mechanisms adapted from Transformers to weigh important inputs.
* **Dual Heads**: Classification + Reconstruction for multitask learning.
* **Time-Series Modeling**: LSTM decoder to capture sequential behavior.
* **Optimization**: Uses `polars`, `Numba`, `PyTorch`, GPU parallelism, and lazy evaluation to significantly improve performance.
* **Explainability**: SHAP and t-SNE provide transparency for financial decisions.

---

## Results

| Model                    | AUC Score   |
| ------------------------ | ----------- |
| Logistic Regression      | 0.71739     |
| Decision Tree            | 0.72383     |
| Random Forest            | 0.74657     |
| XGBoost                  | 0.76869     |
| Fully Connected NN       | 0.76908     |
| Wide & Deep NN           | 0.75824     |
| **FA-GNN + LSTM (Ours)** | **0.77067** |

---

## Performance Optimization

| Pipeline Stage        | Speedup | Techniques Used                         |
| --------------------- | ------- | --------------------------------------- |
| Data Loading          | 19.3×   | `polars`, parallel I/O                  |
| Dataset Preprocessing | 2.2×    | PCA, parallel K-Means, SMOTE, `Numba`   |
| Model Training        | 1.6×    | GPU-accelerated PyTorch, early stopping |
| Graph Visualization   | 1.9×    | `Numba`, `multiprocessing`, `itertools` |

---

## Model Architecture

* **Feature Inputs**: Numerical and categorical features processed separately.
* **Attention Layers**: Capture important patterns with multi-head self-attention.
* **GNN Layers**: Learn borrower relationships with edge similarity and thresholding.
* **LSTM Decoder**: Models time-varying borrower behavior.
* **Adaptive Fusion**: Balances GNN and attention-based representations.

![image](https://github.com/user-attachments/assets/c171eca9-b4ef-4930-95fe-2c37232202c2)

---

## Explainability

* **SHAP Values**: Identify top contributing features (e.g., credit history, employment days).
* **Graph Evolution**: t-SNE visualizations show how borrower clusters evolve over training.
* **Risk Stratification**: Visual labels by predicted default probability (low, medium, high).

---
## Dataset

We use the publicly available [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk). Ensure it is downloaded and placed in the appropriate `data/` directory.

---

## Tech Stack

* Python (Polars, Numba, PyTorch, Scikit-learn)
* CUDA / GPU Acceleration
* SHAP, t-SNE for interpretability
* Lazy graph evaluation and batched GPU loaders
