import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from encoder import Encoder
from decoder import Decoder


def sample_z(mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, x):
        mean, log_var = self.encoder(x)
        z = sample_z(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var
    
def vae_loss(x, reconstruction, mean, log_var):
    # Reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(x, reconstruction)
    )
    reconstruction_loss *= 28 * 28  # Scale up for entire image
    # KL Divergence
    kl_divergence = -0.5 * tf.reduce_sum(
        1 + log_var - tf.square(mean) - tf.exp(log_var),
        axis=1
    )
    kl_divergence = tf.reduce_mean(kl_divergence)

    return reconstruction_loss + kl_divergence


@tf.function
def train_step(vae, x, optimizer):
    with tf.GradientTape() as tape:
        reconstruction, mean, log_var = vae(x)
        loss = vae_loss(x, reconstruction, mean, log_var)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

