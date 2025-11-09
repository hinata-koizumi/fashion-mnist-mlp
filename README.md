# University of Tokyo Deep Learning Course Competition

English | [日本語版(Japanese)](README.ja.md)

## Competition Results
- **Final Rank**: **11th** out of 1,466 participants
- **LB Score**: **0.9204**

## Overview
Classification of Fashion MNIST (10 classes), a fashion version of MNIST, using a Multi-Layer Perceptron.

For more details about Fashion MNIST, please refer to the following link:
Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## Rules
- Training data is provided as x_train, t_train, and test data as x_test.
- Prediction labels should be represented as class labels 0~9, not as one-hot encoding.
- Do not use training data other than x_train and t_train specified in the cells below.
- The MLP algorithm must be implemented using only NumPy (do not use sklearn, tensorflow, etc.).
- It is acceptable to use sklearn functions for data preprocessing (e.g., sklearn.model_selection.train_test_split).

## Approach

- Data Preprocessing/Splitting
  - Load `x_train.npy`, `y_train.npy`, `x_test.npy` and flatten 28×28 images to `float32` (784 dimensions)
  - Split last 10% of training data for validation (fixed seed, default `seed=42`)
  - Standardize all data using mean/std from training split (per feature, add 1e-7 to std to avoid division by zero)
  - Minibatches are shuffled only during training; validation/test use fixed order

- Image Augmentation (Scheduled)
  - Translation: Maximum ±2px (up/down/left/right, implemented with `np.roll`, boundaries zero-padded)
  - Cutout: Square masks (2~4px) at random positions
  - Gaussian noise: σ=0.02 (only when `--enable_noise_aug` flag is set, probability decays from 0.2→0.1)
  - Application probability schedule: Before 80% progress, translation p=0.7, Cutout p=0.20. After 80%, translation p linearly decreases from 0.7→0.4, Cutout p from 0.20→0.08. Cutout stops after 90%
  - Augmentation is applied to raw pixels before standardization, then re-standardized

- Features
  - Raw pixels only: 28×28 raw pixels flattened to 784 dimensions (no external features like HOG)
  - Standardization: Per-feature mean/std estimated from training split

- Model (NumPy-implemented MLP)
  - Architecture: Input 784 → Hidden layers (default `--hidden_sizes` 512→256→128) → Output 10
  - Activation: ReLU for intermediate layers, logits for final layer (no softmax)
  - Dropout: Inverted dropout (training only). After 80%, linearly reduced to `--final_dropout_target` (default 0.05)
  - Initialization: He initialization (for ReLU, `sqrt(2.0 / in_dim)`)
  - Optimization: AdamW (default, `--adamw`) or Adam (`--no-adamw`). Decoupled weight decay for AdamW, L2 regularization added to gradients for Adam
  - Gradient clipping: Global norm `--grad_clip` (default 5.0)
  - EMA: Exponential moving average with `--ema_decay` (default 0.9995). Temporarily strengthened to 0.9997 after 90% progress

- Training and Regularization
  - Optimizer: AdamW (`weight_decay` default 1e-4, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`)
  - Learning rate: 5% warmup of total steps, then cosine decay (base LR `--lr` default 1e-3, minimum LR is base LR × 1e-2)
  - Loss: Softmax cross-entropy. Label smoothing (`--label_smoothing_eps` default 0.03) linearly decays to 0 after 80% progress
  - Regularization: AdamW decoupled weight decay (default 1e-4). L2 term (`--l2` default 1e-4) added to gradients when Adam is selected
  - Gradient clipping: Global norm 5.0
  - Early stopping: Stop if validation accuracy doesn't improve for `--patience` epochs (default 15). Keep best epoch parameters
  - SWA (optional): Snapshot EMA weights in latter half of training (after `--swa_start_frac` default 0.8, every `--swa_interval` epochs), average them at the end and set to EMA
  - Fine-tuning (optional): Fine-tune on full training data for `--ft_epochs` (default 2) epochs after training (LR × 0.2, Dropout × 1/3, label smoothing 0)

- Validation and Selection (Inference Selector)
  - Validation inference: Evaluate accuracy with MC Dropout (n=30, temperature `--mc_temp` default 1.0) for probability stabilization. Record confusion matrix, per-class Precision/Recall/F1, calibration metrics (ECE, entropy, margin) every 5 epochs
  - Candidate strategies: `det` (deterministic inference with EMA weights), `mc@T` (explore temperature T∈{0.75,0.80,0.85,0.90,0.95,1.00} and dropout override∈{None,0.15,0.12,0.10,0.075}), `tta5mc@T` (5-direction logit averaging with MC), `tta9mc@T` (9-direction logit averaging with MC), `tta5_center` (5-direction probability averaging, center weight 2.0, no MC)
  - Selection method: Adopt candidate with highest validation accuracy. When difference is small (within 3e-4), explore blending of top 2/3 candidates (α∈{0.2..0.8}). When `--enable_adaptive_blend`, optimize α adaptively
  - Multi-seed: Sequentially run multiple seeds with `--multi_seed`, also supports weighted ensemble based on each seed's validation accuracy (accuracy^`--ensemble_weighted_power`)

- TTA and Temperature Scaling
  - TTA: Translation only (no rotation/flip). 5 directions (center + up/down/left/right) or 9 directions (center + 8 neighbors). Swap EMA once and run inference for all directions together for speedup. Logit averaging (9 directions) or probability averaging (5 directions with center weight)
  - Temperature scaling (optional): When `--enable_temp_scale`, grid search T∈{0.85,0.9,0.95,1.0,1.05,1.1} on selected candidate's validation probabilities and apply to test probabilities

- MC Dropout (Automatic)
  - Enable dropout during inference and use average probability of multiple samples (default 30). Select temperature and dropout override on validation. Can be combined with TTA×MC

- Inference/Saving
  - Determine classes from selected strategy's probabilities and save to `data/output/y_pred.csv` with `label` header
  - Model saving: Save as `data/train/mlp_model.npz` including weights/biases, EMA, optimizer state, training settings, standardization mean/std, metadata (best validation accuracy, strategy name)
  - Options: Save test probabilities as `*.npy` with `--save_test_proba`, output selection strategy and validation accuracy to sidecar `*.meta.json`. Support weighted average of multiple probabilities and CSV generation with `--ensemble_from`

## Technologies Used

- Python 3
- NumPy (`numpy`)
- Python Standard Library (`argparse`, `json`, `os`, `sys`, `subprocess`, `typing`)
