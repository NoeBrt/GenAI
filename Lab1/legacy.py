import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

latent_dim = 2  # Latent space dimension

# Reparameterization trick
def sample_z(mean, log_var):
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

# Loss Function
def vae_loss(x, reconstruction, mean, log_var):
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(x, reconstruction)
    )
    reconstruction_loss *= 28 * 28
    kl_divergence = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return reconstruction_loss + kl_divergence

# Encoder
class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

# Decoder
class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(28 * 28, activation="sigmoid")
        self.reshape = layers.Reshape((28, 28, 1))

    def call(self, z):
        z = self.dense1(z)
        z = self.dense2(z)
        return self.reshape(z)

# VAE
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

# Parameters
latent_dim = 2
epochs = 20
batch_size = 128

# Load data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Training step
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        reconstruction, mean, log_var = vae(x)
        loss = vae_loss(x, reconstruction, mean, log_var)
        gradients = tape.gradient(loss, vae.trainable_variables)  # Check this line
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss


# Training
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
for epoch in range(epochs):
    for step, x_batch in enumerate(train_dataset):
        loss = train_step(x_batch)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Visualization
def visualize_reconstruction(vae, x_test, n=10):
    mean, log_var = vae.encoder(x_test)
    z = sample_z(mean, log_var)
    reconstruction = vae.decoder(z)

    # Plotting the original and reconstructed images
    fig, axes = plt.subplots(n, 2, figsize=(10, 2 * n))
    for i in range(n):
        axes[i, 0].imshow(np.squeeze(x_test[i]), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(np.squeeze(reconstruction[i]), cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.show()

# Example visualization after training
encoder = Encoder(latent_dim)
decoder = Decoder()
vae = VAE(encoder, decoder)
visualize_reconstruction(vae, x_test[:10])
