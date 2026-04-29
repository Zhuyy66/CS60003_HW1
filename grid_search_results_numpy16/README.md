# NumPy-only 16x16 Grid Search Results

This directory contains the final strict-NumPy result.  Only `image_size=16`
is used; no 32x32 or 64x64 experiments are included in the final report.

## Grid

- backend: NumPy
- image_size: 16
- hidden_dim: [128, 256, 512]
- learning_rate: [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16]
- lr_decay: [0.9, 0.94, 0.95, 0.97]
- weight_decay: [0.0, 0.0001, 0.0005]
- epochs: 30
- batch_size: 256

## Best Result

- hidden_dim: 512
- learning_rate: 0.16
- lr_decay: 0.9
- weight_decay: 0.0005
- best_epoch: 30
- best_val_accuracy: 0.6859
- test_accuracy: 0.6881
- test_loss: 1.1440
