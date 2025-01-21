# For pedagogic purpose, the code is sequential here, it should be of form : function
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

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