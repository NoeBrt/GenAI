# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from encoder import Encoder
from decoder import Decoder
from vae import VAE, vae_loss, train_step
import argparse


def main(args):
    # Load MNIST dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    latent_dim = args.latent_dim

    optimizer = tf.keras.optimizers.Adam()

    encoder = Encoder(latent_dim)
    decoder = Decoder()

    vae = VAE(encoder, decoder)

    # Training
    epochs = args.epochs
    batch_size = args.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
    for epoch in range(epochs):
        for step, x_batch in enumerate(train_dataset):
            loss = train_step(vae, x_batch, optimizer)
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")


    num_images = args.num_examples
    x_sample = x_test[:num_images]
    # Generate images
    reconstruction, _, _ = vae(x_sample)

    # Display original images
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(x_sample[i, :, :, 0], cmap="gray")
        plt.title("Original")
        plt.axis("off")

    # Display generated images
    for i in range(num_images):
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstruction[i, :, :, 0], cmap="gray")
        plt.title("Generated")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.output_dir + f'/vae_results_{args.latent_dim}_{args.batch_size}.png')
    plt.show()

    # Display latent space
    x_random = tf.random.normal(shape=(num_images, latent_dim))
    reconstruction = decoder(x_random)

    # Display original random latent vectors
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(tf.expand_dims(x_random[i], axis=0).numpy(), cmap="gray")
        plt.title("Latent Vector")
        plt.axis("off")

    # Display generated images
    for i in range(num_images):
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstruction[i, :, :, 0].numpy(), cmap="gray")  
        plt.title("Generated")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.output_dir + f'/gan_results_{args.latent_dim}_{args.batch_size}.png')
    plt.show()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and visualize VAE on MNIST')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimension of latent space')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--output_dir', type=str, default='vae_results', help='Output directory for visualizations')
    
    args = parser.parse_args()
    main(args)