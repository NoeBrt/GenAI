# GANInterface.py
from abc import ABC, abstractmethod

class GANInterface(ABC):
    @staticmethod
    @abstractmethod
    def build_generator(latent_dim=100):
        """Return a Keras model for the generator."""
        pass

    @staticmethod
    @abstractmethod
    def build_discriminator():
        """Return a Keras model for the discriminator."""
        pass

    @staticmethod
    @abstractmethod
    def build_gan(discriminator, generator, latent_dim):
        """Combine the generator and discriminator into a GAN model."""
        pass

    @staticmethod
    @abstractmethod
    def train(x_train, generator, discriminator, gan, epochs, batch_size, latent_dim):
        """Train the GAN."""
        pass
