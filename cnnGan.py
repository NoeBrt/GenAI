# cnnGan.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from GANInterface import GANInterface

class cnnGan(GANInterface):
    
    @staticmethod
    def build_generator(latent_dim=100):
        model = tf.keras.Sequential([
            layers.Dense(7 * 7 * 256, input_dim=latent_dim),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(1, kernel_size=7, activation="tanh", padding="same")
        ])
        return model

    @staticmethod
    def build_discriminator():
        model = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(28, 28, 1)),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    @staticmethod
    def build_gan(discriminator, generator, latent_dim):
        gan_input = layers.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        gan = tf.keras.Model(gan_input, gan_output)
        return gan

    @staticmethod
    def train(x_train, generator, discriminator, gan, epochs, batch_size=128, latent_dim=100):
        for epoch in range(epochs):
            for _ in range(x_train.shape[0] // batch_size):
                # Train discriminator with real images
                noise = tf.random.normal([batch_size, latent_dim])
                fake_images = generator.predict(noise)
                real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
                
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                
                discriminator.trainable = True
                d_loss_real = discriminator.train_on_batch(real_images, real_labels)
                d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
                discriminator.trainable = False

                # Train generator (via GAN, where discriminator is not trainable)
                misleading_labels = tf.ones((batch_size, 1))
                g_loss = gan.train_on_batch(noise, misleading_labels)

            print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_real[0] + d_loss_fake[0]}, G Loss: {g_loss}")
