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

## Code Architecture

- **`cnnGan.py`**: Defines the CNN-based GAN architecture (Generator and Discriminator).
- **`transformerGan.py`**: Implements the Transformer-based GAN with MHSA.
- **`GANInterface.py`**: Static interface that define gan classic functions.
- **`commonGan.py`**: Contains utility functions for data loading and image generation.
- **`main.py`**: The entry point for training and generating images.

---

## CNN Gan

## Transformer Gan

## Results 

### CNN results with 10 epochs
![cnn_gan_10epoch](https://github.com/user-attachments/assets/4c02a047-2f83-4da1-890c-3d790ac3567e)

### Transformers results with 10 epochs
![tranformer_10_epochs2](https://github.com/user-attachments/assets/f5432d6a-d9a3-4587-a494-40a76fd97f0d)


## FAQs

### 1. What is Transpose Convolution, and why do we use it in the Generator?
According to keras documentation, the tranpose convolution corespond to the opposite of a convolution (also called a deconvolution or upsampling), in our generator on the CNN GAN, it allow to retrieve the dimension of a 28\*28 image from our latent vector (after a prokection in a 7\*7\*256 space and a reshaping to a (7,7,128) tensor).

### 2. Why Use LeakyReLU and Sigmoid in the Discriminator?

- **LeakyReLU**: Prevents the vanishing gradient problem by allowing a small gradient for negative values.

LeakyReLU is similar to ReLU but assigns a small slope for negative inputs instead of zero. This ensures that neurons continue to update during training rather than getting "stuck" with zero gradients. In our discriminator, we apply two LeakyReLU activations (with an alpha of 0.2) after each convolution, maintaining gradient flow and non-linearity so the model can learn complex patterns.

![image](https://github.com/user-attachments/assets/e2c454e4-f447-4557-88a0-9e30a55aebab)



- **Sigmoid**:After the convolutional layers, a sigmoid function converts the output into a probability score between 0 and 1. During the training we want that the weights lead the input to the right probabilities.

![image](https://github.com/user-attachments/assets/8381b8e3-7519-4f06-b620-a9f011972a5b)

---
