from cnnGan import cnnGan
from tranformerGan import transformerGan
import commonGan
import argparse
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from GANInterface import GANInterface

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected by TensorFlow:")
    for gpu in gpus:
        print("  ", gpu)
else:
    print("No GPU detected by TensorFlow. Running on CPU.")

print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))


gan_dict=  {
    "cnn": cnnGan,
    "transformer": transformerGan

}


def main():
    parser = argparse.ArgumentParser(description='GAN Training Script')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    parser.add_argument('--dataset', type=str, default=None, help='dataset to use')
    parser.add_argument('--model', type=str, default="cnn", help='model to use')
    parser.add_argument('--output', type=str, default="output.png", help='output file')
    parser.add_argument('--n_images', type=int, default=10, help='number of images to generate')
    args = parser.parse_args()

    learning_rate=args.learning_rate
    latent_dim=args.latent_dim
    epochs=args.epochs
    batch_size=args.batch_size
    n_images=args.n_images

    if args.model not in gan_dict:
        print("Invalid model")
        return

    gan_class= gan_dict[args.model]

    x_train=commonGan.load_data()
    generator = gan_class.build_generator()
    print(generator.summary())
    discriminator = gan_class.build_discriminator()
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    discriminator.trainable = False
    gan=gan_class.build_gan(discriminator, generator, latent_dim)
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy")
    gan_class.train(x_train,generator, discriminator, gan, epochs, batch_size,latent_dim)
    commonGan.generate_images(generator, n_images,latent_dim)








if __name__ == '__main__':
    main()
