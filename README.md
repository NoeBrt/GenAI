# GAN Training and Comparison: CNN-Based vs Transformer-Based

## Overview
This repository implements and compares two Generative Adversarial Network (GAN) models:
1. **CNN-based GAN**
   - Uses Convolutional Neural Networks (CNNs) for both the Generator and Discriminator.
2. **Transformer-based GAN**
   - Utilizes Transformer architecture with Multi-Head Self-Attention (MHSA) for the Generator and Discriminator.

The models are trained on the **MNIST dataset**, and their performance is compared based on the quality of generated images.

---

## Table of Contents
1. [Instructions](#instructions)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Explanation](#code-explanation)
5. [FAQs](#faqs)
6. [License](#license)

---

## Instructions

### Part 1: CNN-based GAN
1. **Generator**
   - Uses Transpose Convolution layers to upsample noise into images.
   - Activation: ReLU for hidden layers, Tanh for output layer.

2. **Discriminator**
   - Uses Convolution layers to downsample input images.
   - Activation: LeakyReLU for hidden layers, Sigmoid for output layer.

3. **Training**
   - The model is trained on the MNIST dataset to generate handwritten digits.

**Questions:**
1. What is Transpose Convolution, and why do we use it in the Generator?
2. What are LeakyReLU and Sigmoid, and why are they used in the Discriminator?

---

### Part 2: Transformer-based GAN
1. **Generator**
   - Uses Multi-Head Self-Attention (MHSA) and positional encodings.
   - Upsamples the latent space using feedforward layers.

2. **Discriminator**
   - Analyzes global relationships using MHSA.
   - Classifies real vs fake images.

3. **Training**
   - Trained on the MNIST dataset and compared with the CNN-based GAN.

---

## Installation

### Install Required Packages
```bash
pip install -r requirements.txt
```

---

## Usage
To train a GAN model:

### Train the CNN-based GAN
```bash
python main.py --model cnn --epochs 100 --batch_size 64 --learning_rate 0.0002 --latent_dim 100
```

### Train the Transformer-based GAN
```bash
python main.py --model transformer --epochs 100 --batch_size 64 --learning_rate 0.0002 --latent_dim 100
```

## Code Explanation

- **`cnnGan.py`**: Defines the CNN-based GAN architecture (Generator and Discriminator).
- **`transformerGan.py`**: Implements the Transformer-based GAN with MHSA.
- **`commonGan.py`**: Contains utility functions for data loading and image generation.
- **`main.py`**: The entry point for training and generating images.

---

## Results 

### CNN results with 10 epochs
![cnn_gan_10epoch](https://github.com/user-attachments/assets/4c02a047-2f83-4da1-890c-3d790ac3567e)

### Transformers results with 10 epochs
![tranformer_10_epochs2](https://github.com/user-attachments/assets/f5432d6a-d9a3-4587-a494-40a76fd97f0d)


## FAQs

### 1. What is Transpose Convolution, and why do we use it in the Generator?
Transpose Convolution is used to upsample the input (latent vector) to an image size. It helps create high-resolution outputs.

### 2. Why use LeakyReLU and Sigmoid in the Discriminator?
- **LeakyReLU**: Prevents the vanishing gradient problem by allowing a small gradient for negative values.
- **Sigmoid**: Outputs a probability indicating whether the image is real or fake.

---
