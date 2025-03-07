# Lab 4 -  Introduction to Stable Diffusion Models

## Part 1: Preliminary Activity - Neural Network for Function Inversion

In this part we try to approximate the inverse of $x=\sin(y)$, so $y=arcsin(x) $  

### Dataset

We use a uniform distribution of random number between -1 and 1 as a X and $y = sin(X)$

```X = np.random.uniform(-1, 1, n) ```
```y = np.sin(X)```

### Model

I used 3 model

1. Single ReLU
2. 
   * 1 input neuron (for y),
   * 1 hidden layer (3 neurons, ReLU activation)
   * 1 output neuron (for x).
  
3. Double ReLU
   * 1 input neuron (for y),
   * 2 hidden layer (3 neurons, ReLU activation)
   * 1 output neuron (for x).

4. simple linear
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

3. Single Linear
![alt text](arcsin_linear.png)


The double relu have the lowest MSE.

The 3 models estimate well the line between $[-0.7,0.7]$ but struggle to estimate the curve of the arcsin, it struggle to estimate the non-linear part of arcsin. It coulb be also linked to the distribution, maybe their is not enought data on the hight curvature of sin.

## Key Discussion Points:


### What happens for values outside the range [-1,1]?

When we predict values outside $[-1,1]$ the model stay linear.

![alt text](outside_range.png)


When we train values outside $[-1,1]$ the model lost performance even on the linear part, $arcsin$ is on $[-1,1]$, it's mostly because the model is noisier to do the inverse.

![alt text](outside_range_training.png)


### What are the implications of approximating inverses in more complex functions ?

On more complex function we have similar challenge :

* Unpredictability outside the domain.
* Harder prediction on non-monotonous function (so harder on complex functions).


# Part 2: Diffusion Models on Images

 