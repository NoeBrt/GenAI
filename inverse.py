import os
# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Option 1: Disable GPU if you don't need it (uncomment the next line)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def generate_dataset(n=1000):
    np.random.seed(0)
    X = np.random.uniform(-1, 1, n)
    y = np.sin(X)
    return X, y

def build_model():
    model = Sequential([
        Input(shape=(1,)),
        Dense(3, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model


def train_model(model, X, y, n=1000):
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X, y, epochs=n, callbacks=[es])
    return model, history


def visualize(model):
    X_linspace = np.linspace(-1, 1, 1000)
    y_pred = model.predict(X_linspace)
    y_test = np.sin(X_linspace)
    test_loss=model.evaluate(X_linspace, y_test)  # evaluate the model
    print("Test loss: ", test_loss)
    plt.plot(X_linspace, y_pred, color='red', label='Model Prediction x')
    y_sin = np.arcsin(X_linspace)
    plt.plot(X_linspace, y_sin, color='green', linestyle='-', label='true arcsin')
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Model Approximation of arcsin(y) - test loss = ' + str(test_loss))
    plt.show()
    
    
def main():
    X, y = generate_dataset(100)
    model = build_model()
    model, history = train_model(model, y, X)
    visualize(model)
    model.save('model.h5')
    print('Model saved as model.h5')

    
if __name__ == '__main__':
    main()
