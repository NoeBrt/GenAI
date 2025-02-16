# GAN Training and Comparison: CNN-Based vs Transformer-Based

## Overview
This repository implements and compares two Generative Adversarial Network (GAN) models:

**What is a GAN?**

A GAN is a type of network architecture that consists of two neural network: a Generator, wich purpose is to generate new data from a latent space, and a Discriminator, trained in parallel to *discriminate* between real and fake data.

**How is it trained?**

The training with data of a GAN :
1. The Generator generates a batch of fake data from a latent space.
2. we attribute a label of 0 to the fake batch 1 to a batch of real data.
3. The Discriminator is trained to distinguish between real and fake data.
4. We froze the weight update of the Discriminator and we train the Generator to generate data that the Discriminator will classify as real. By creating a batch of fake data (of the latent space dimension using a distribution like normal distribution) and attributing a label of 1 to it. We minimise the loss of the Generator so the output of the Discriminator is close to 1 for generated data.

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
A Convolutional Neural Network (CNN) in the context of GANs is a type of neural network that uses convolutional layers to extract features from images. In our CNN-based GAN, we use CNNs for both the Generator and Discriminator.

### Generator

```python
        model = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 256),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(1, kernel_size=7, activation="tanh", padding="same")
        ])
```

**What is used in the model ?**
- ***Sequential*** : A Sequential model i for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
- ***Input*** : A input layer is used to define the input shape of the model.
- ***Dense*** : A dense layer permit the model to learn non-linear relationships in the data by using a set of weights and biases to change the size of the entry vector.
   - ***filters***
   - ***kernel_size*** : The kernel size is the size of the filter matrix for the convolution.
   - ***strides*** : The stride is the number of pixels by which the filter matrix is shifted over the input matrix.
   - ***padding*** : Padding is a technique used to preserve the spatial dimensions of the input volume.
   - ***activation*** : Activation function is at the of each layer, it's used to introduce non-linearity to the model.
- ***Reshape*** : Reshape layer is used to change the shape of the input tensor.
- ***Conv2DTranspose*** : Conv2DTranspose layer is used to upsample the input tensor, it's the opposite of a convolutional layer.


**steps**
1. The input layer takes a latent vector as input.
   ```python
   layers.Input(shape=(latent_dim,))
   ```

2. The dense layer projects the latent vector into a 7x7x256 tensor. (the model will learn how to extand the vector)
   ```python
   layers.Dense(7 * 7 * 256)
   ```

3. The reshape layer changes the shape of the tensor to (7, 7, 256) (like a 7x7 image with 256 channels).
   ```python
   layers.Reshape((7, 7, 256))
   ```

- **Conv2DTranspose Layer Details:**

  - **First Conv2DTranspose Layer:**
    ```python
    layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu")
    ```
    - **Number of Filters:** 128
      * Each filter is designed to detect a unique feature in the input.
    - **Kernel Size:** 4x4
      * This defines the spatial dimensions of the filter that slides over the input tensor.
    - **Stride:** 2
      * The filter moves 2 pixels at a time, effectively increasing the spatial dimensions (upscaling) of the output.
    - **Padding:** "same"
      * Padding is applied to conserve the spatial dimensions after the convolution operation.

  - **Second Conv2DTranspose Layer:**
    ```python
    layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")
    ```
    - **Number of Filters:** 64
      * Further refines the features learned from the previous layer.
    - **Kernel Size:** 4x4
      * Defines the spatial dimensions of the filters.
    - **Stride:** 2
      * Further increases the spatial dimensions.
    - **Padding:** "same"
      * Ensures the output maintains the correct dimensions.

  - **Final Conv2DTranspose Layer:**
    ```python
    layers.Conv2DTranspose(1, kernel_size=7, activation="tanh", padding="same")
    ```
    - **Number of Filters:** 1
      * Outputs a single-channel image (grayscale).
    - **Kernel Size:** 7x7
      * Covers the entire spatial structure of the input tensor.
    - **Activation:** "tanh"
      * Maps the output values between -1 and 1 to correspond to pixel values in the image.
    - **Padding:** "same"
      * Ensures the final output maintains the right spatial dimensions.

### Discriminator
```python
        model = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
```

## Transformer Gan
```python
        model = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 256),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(1, kernel_size=7, activation="tanh", padding="same")
        ])
```

### Discriminator
```python
        model = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
```
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
