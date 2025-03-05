# Lab 4 -  Introduction to Stable Diffusion Models

## Part 1: Preliminary Activity - Neural Network for Function Inversion

In this part we try to approximat the inverse of $x=\sin(y)$, so $y=arcsin(x) $  

### Dataset

We use a uniform distribution of random number between -1 and 1 as a X and $y = sin(X)$

```X = np.random.uniform(-1, 1, n) ```
```y = np.sin(X)```

### Model

I used 3 model

1. Single ReLU
   * 1 input neuron (for y),
   * 1 hidden layer (3 neurons, ReLU activation)
   * 1 output neuron (for x).
  
2. Double ReLU
   * 1 input neuron (for y),
   * 2 hidden layer (3 neurons, ReLU activation)
   * 1 output neuron (for x).

2. simple linear
   * 1 input neuron (for y),
   * 1 hidden layer (3 neurons, linear activation)
   * 1 output neuron (for x).

* Loss MSE

### Training 

```model = train_model(model, y, X) ```
training parameter :
* trained on y 
* 1000 epochs with early stopping on loss

### Evaluation

1. Single ReLU
![alt text](arcsin_relu1.png)

2. Double ReLU
![alt text](arcsin_relu2.png)

2. Single Linear
![alt text](arcsin_linear.png)
