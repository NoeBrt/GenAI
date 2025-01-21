# Variational Autoencoder (VAE) Implementation for MNIST

This project implements a Variational Autoencoder (VAE) to generate and reconstruct MNIST digits using TensorFlow 2.x.

## Overview

The VAE consists of an encoder that compresses images into a lower-dimensional latent space and a decoder that reconstructs images from this latent representation. This implementation includes:

- Encoder network that maps images to mean and variance of latent distribution
- Decoder network that reconstructs images from latent vectors
- Training loop with reparameterization trick and KL divergence loss
- Visualization of original vs. reconstructed images
- Generation of new images from random latent vectors

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib

## Project Structure

```
.
├── encoder.py     # Encoder network implementation
├── decoder.py     # Decoder network implementation
├── vae.py        # VAE model and training utilities
└── main.py       # Training script and visualization
```

## Usage

Run the training script with default parameters:

```bash
python main.py
```

Or customize the training with command-line arguments:

```bash
python main.py --latent_dim 10 --batch_size 256 --epochs 50 --num_examples 10 --output_dir results
```

### Command-line Arguments

- `--latent_dim`: Dimension of the latent space (default: 2)
- `--batch_size`: Batch size for training (default: 128)
- `--epochs`: Number of training epochs (default: 20)
- `--num_examples`: Number of examples to visualize (default: 5)
- `--output_dir`: Output directory for saving visualizations (default: 'vae_results')

## Output

The script generates two visualization plots:

1. `vae_results_{latent_dim}_{batch_size}.png`: Shows original MNIST digits and their VAE reconstructions
2. `gan_results_{latent_dim}_{batch_size}.png`: Shows random latent vectors and their corresponding generated digits

example:

`vae_results_2_128.png`:
![vae_results_2_128](https://github.com/user-attachments/assets/07af1bab-9753-45c3-b3c6-02b0504768ea)

`gan_results_2_128.png`:
![gan_results_2_128](https://github.com/user-attachments/assets/01099911-0301-41ff-91ce-ebdc983061a6)

## Model Architecture

### Encoder
- Flattens input images
- Multiple dense layers with ReLU activation
- Outputs mean and log variance for latent space

### Decoder
- Takes latent vectors as input
- Multiple dense layers with ReLU activation
- Final layer reconstructs image with sigmoid activation
- Reshapes output to match original image dimensions

## Training Process

1. Load and preprocess MNIST dataset
2. Initialize encoder and decoder networks
3. Train for specified number of epochs:
   - Compute reconstruction loss and KL divergence
   - Update model weights using Adam optimizer
4. Generate visualizations of results

## Example Results

The visualizations show:
- Original MNIST digits compared to their reconstructions
- Random latent vectors and their generated digits
- Quality of reconstruction depends on latent dimension size

## Notes

- Higher latent dimensions generally lead to better reconstruction quality
- Lower latent dimensions may capture more compressed representations
- Training time increases with batch size and number of epochs
