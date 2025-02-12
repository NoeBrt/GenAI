import tensorflow as tf
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def generate_images(generator, n_images,latent_dim):
    noise = tf.random.normal([n_images, latent_dim])
    generated_images = generator.predict(noise)
    fig, axes = plt.subplots(1, n_images, figsize=(20, 4))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.show()


def load_data(custom_dataset=None):
    if custom_dataset is None:
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    else:
        (x_train, _) = custom_dataset

    x_train = x_train.astype("float32") / 255.0
    # Scale images to [-1, 1] instead of [0, 1]
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
    # Now the images are in [-1, 1]
    return x_train
