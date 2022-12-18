# Week 1

Learning the practical aspects of how to make your neural network work well. Ranging from things like hyperparameter tuning to how to set up your data, to how to make sure your optimization algorithm runs quickly so that you get your learning algorithm to learn in a reasonable amount of time.

## Train / Dev / Test sets

Making good choices in how you set up your training, development, and test sets can make a huge difference in helping you quickly find a good high-performance neural network.

1. How many layers

2. How many hidden units

3. Learning rates

4. Activation functions

It's almost impossible to correctly guess the right values for all of these, and for other hyperparameter choices, on your first attempt. So, in practice, applied machine learning is a highly iterative process, in which you often start with an idea, such as you want to build a neural network of a certain number of layers, a certain number of hidden units, maybe on certain data sets, and so on. And then you just have to code it up and try it, by running your code. You run an experiment and you get back a result that tells you how well this particular network, or this particular configuration works. And based on the outcome, you might then refine your ideas and change your choices and maybe keep iterating, in order to try to find a better and a better, neural network.

- We use dev set to see which **model fits better on our data** and then we use that model to evaulate our test set.

- Dev set size needs to be only big enough for us to evaluate our model.

- When we have a large amount of data, dev and test sets portion get smaller.

- Mismatch train/test distribution

- Make sure dev and test sets come from same distribution

- Not having a test set might be okay (Only dev set) (the goal of the test set is to give you a ... unbiased estimate of the performance of your final network, of the network that you selected. But if you don't need that unbiased estimate, then it might be okay to not have a test set. (overfitting to dev set))

## Bias / Variance

- High bias = underfitting

- High variance = overfitting

In high dimensional we cannot plot data so we use different classifiers to classify bias and variance for us:

1. Train set error

2. Dev set error

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/Train:Dev%20error.png" alt="" width="520" data-align="center">

High bias and High variance : 

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/High%20bias-variance.png" alt="" width="257" data-align="center">

## Basic recipe for Machine Learning

- High bias ?  (Training data performance) -> bigger network, train larger, NN architecture

- High variance ? (Dev set performance) -> More data, Regularization, NN architecture

- Low bias and Low variance (DONE)

In the pre-deep learning era, we didn't have as many tools that just reduce bias, or that just reduce variance without hurting the other one. But now we can only reduce one without hurting the other one. If we use regularization there is a little bit tradeoff between bias and variance.

## Regularization

**Logistic regresstion**

- L2 Regularization
  
  <img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/L2.png" alt="" width="429" data-align="center">
  
  Since w is a matrix and has more values, it might overfit more but B is only a real number.

- L1 Regularization
  
  Some people say that this can help with compressing the model, because 
  
  the set of parameters are zero, then you need less memory to store the model.
  
  <img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/L1.png" title="" alt="" data-align="center">

- Regularization parameter 
  
  We usually define this using cross validation set. Another hyperparameter we need to tune.

**Neural network**

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/neural%20network.png)

- Forbenius norm of a matirx

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/formula.png" title="" alt="" data-align="center">

- Weight decay
  
  ![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/weight-decay.png)

## Why regularization reduces overfitting?

1. If we put our regularization lambda really really big, that makes the weight of matrixes to be reasonably close to zero. So a lot of hidden units thats basically zeroing out a lot of the impact of there hidden units and we have a simpler neural network.

2. We know in activation functions Z = W * a + b, so if our regularization parameter is large, the paramter w be samll and we end up to linear regression.

## Dropout regularization

With dropout, what we're going to do is go through each of the layers of the network and set some probability of eliminating a node in neural network. In each training example we diminish the network by eliminating a node randomly.

- Inverted Dropout
  
  when we reduce **a** values for example by 20 percent chances it will cause **Z** values to change as well and the expected result to change. To prevent this effect we devide **a** by  0.8.
  
  We dont use drop out in test sets. because we are making predictions and we dont want to be it random but this inverted Dropout makes the test sets easier because we have a less of scaling problem and make us sure the expected result wont change.

## Undrestanding Dropout

- The reason why drop out works is it cant rely on any one feature (input of a node), since any of them can be removed. So we spread out the weights in all of the inputs. This event is called shrink weights and its the same as L2 regularization.

- We can use differnet keep-prob for each layer.

- For layers we dont care about overfitting we can put keepprob = 1 and we are using all of the units.

- We 99% of time use keep-prob =1 in input layers and output layers.

- One of the downside of dropout is the cost function J is no longer well defined on every iteration.

## Other regularization methods

#### Data augmentation

If we are having overfitting problem and getting more training data is expensive, we can use **DATA AUGMENTATION**.

These are <u>dependent</u> examples, but they are cheaper.

#### Early stopping

Plot either training error or Cost function (both of them must be descreaing) with dev set error and when the dev set start to increase, we should stop it there because there is a place we are facing to overfitting.

When we start our algorithm w is initiazlied with values = 0 and in the end it has a very big values, and in the mid-size rate w there is a place the dev-set error will start to increase.

The problem of early stopping is violating orthogonalization, because it wont let each task to complete all of its work and also we are optimizing cost function and not overfitting at the same time.

#### Orthogonalization

In machine learning we have different type of tasks for example:

- Optimizing cost function J:
  
  1. Gradient descent
  
  2. RMS prob
  
  3. Adam

- Not overfit:
  
  1. Regularization
  
  2. Data augmentation
  
  3. Getting more data

orthogonalization is the principle that each task is seperated from the other task in machinelearning. (One task at a time)

Many people use L2 regularization instead of early stopping, the only disadvantage for L2 regularization that we should compute for different values of lambda and its computionally expensive.

## Normalizing inputs

Use same Mu and sigma we use for normalizing input on test set as well.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/normalizing-input.png)

If we normalize oue input, we may need to have smaller learning rate as well. and we go forward to minimum and makes cost function to minimze and optimize easier.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/unnormalized.png)

## Vanishing / Exploding Gradients

One of the problems of training neural network, is data vanishing and exploding gradients. What that means is that when you're training a very deep network your derivatives or your slopes can sometimes get either very, very big or very, very small, maybe even exponentially small.

If W[l] > Identity metrics the activations can explode. (increase explonantionally)

If W[I] < identity metrics the activations can vanish. (decrease explonantionally) (gradient descent takes small step and takes much more time learning algorithm)

## Weight initialization for Deep networks

A partial solution for vanishing / exploding gradients :

Z = w1x1 + w2x2 + ... + wnxn

The larger n is the smaller wi we want to be because it makes Z so big and if z gets so big a = g(z) so activation become also big and we will explode.

One reasonale thing to do would be set the variance of W to be euqal 1/n. If we are using relu activation function we should use 2/n. ( n in number of input feature going in to a neuron)

We call this process **Xavier initialization.**

We can tune variance as another hyperparameter.

## Numerical Apprroximation of Gradients

When you implement back propagation you'll find that there's a test called  creating checking that can really help you make sure that your implementation of back prop is correct. if approximate error is really small thats  a really good sign we are having a good implementation.

## Gradient checking

1. Take W[1], b[1], ... , W[L], b[L] and reshape into a big vector Theta.

2. Take dW[1], db[1], ..., dW[L], db[L] and reshape into a big vector dTheta.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/graident-checking.png)

for checking d approximation and d theta:

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/check.png" alt="" width="455" data-align="center">


