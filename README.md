# Convolutional Variational Autoencoder (CVAE)

![image alt]()

A PyTorch implementation of a Convolutional Variational Autoencoder for image generation and reconstruction, specifically designed for face datasets.

## Features

- **Convolutional Architecture**: Encoder-decoder structure with convolutional and transposed convolutional layers
- **Perceptual Loss**: VGG19-based perceptual loss for improved reconstruction quality
- **Beta Scheduling**: Gradual increase of KL divergence weight during training
- **Checkpointing**: Resume training from saved checkpoints
- **Memory Efficient**: Built-in memory management for GPU training

## Model Architecture

- **Input**: 128×128 RGB images
- **Encoder**: 3 convolutional blocks (32→64→128 channels) with LayerNorm
- **Latent Space**: 32-dimensional latent vector
- **Decoder**: 3 transposed convolutional blocks with skip connections

## Loss Components

1. **Reconstruction Loss**: Binary Cross-Entropy + MAE
2. **KL Divergence Loss**: Regularization term with beta scheduling
3. **Perceptual Loss**: VGG19 feature matching (optional, enabled after epoch 10)

## Quick Start

```python
from vae_model import ConvolutionnalVAE
from training_config import training_config

# Initialize model
model = ConvolutionnalVAE(
    image_channels=3, 
    z_dim=32, 
    input_size=128
)

# Train model
python train_vae.py
```

## Training Configuration

Key parameters in `training_config`:
- `batch_size`: 32
- `lr`: 1e-4
- `num_epochs`: 100
- `beta`: KL weight (starts at 0.8, increases every 10 epochs)
- `z_dim`: Latent dimension (32)

## Requirements

- PyTorch >= 1.9
- torchvision
- matplotlib
- tqdm

## File Structure

```
├── vae_model.py      # Model architecture
├── losses.py        # Loss functions and VGG19 implementation
├── train_vae.py     # Training script
├── dataset.py       # Dataset loading utilities
└── training_config.py  # Configuration parameters
```

## Features

- **Progressive Training**: Perceptual loss introduced after initial epochs
- **Gradient Clipping**: Prevents gradient explosion
- **Automatic Checkpointing**: Saves model every 10 epochs
- **Training Visualization**: Loss curves and sample outputs
