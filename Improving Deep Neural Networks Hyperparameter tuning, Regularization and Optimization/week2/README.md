# Week 2

Having fast optimization algorithm can speed up training on large data and efficiency of our team.

## Mini batch gradient descent

Vectorization allows you to efficiently compute on m examples, but if m is still big it is still slow.

It turns out that you can get a faster algorithm if you let gradient descent start to make some progress even before you finish processing your entire giant training set.

1. Split your training to smaller training set = mini batches

2. Mini batch t : ( X {t} , Y {t} )

Batch gradient descent = when we process our entire training set all at the same time.

Mini batch gradient descent = we split our training set to small batches.

One epoch = pass through training set.

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/mini-batch-gradient-descent.png" alt="" width="546" data-align="center">

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/batchvsmini.png" alt="" width="514" data-align="center">

- If mini-batch size = m ---> batch gradient descent ( **To long per iteration** )

- If mini-batch size = 1 -----> stochastic gradient descent (every example is a mini-batch but we **lose our speed up from vectorization**)

- If mini-batch size = in between (1-m) ----> mini-batch gradient descent (Fastest learning) (Vectorization + make process without processing entire training set)

Then batch gradient descent might start somewhere and be able to take relatively low noise, relatively large steps. And you could just keep matching to the minimum.

Stochastic gradient descent can be extremely noisy. ( **we can reduce it by reducing learning rate**  ) and on average, it'll take you in a good direction, but sometimes it'll head in the wrong direction as well. As stochastic gradient descent won't ever converge, it'll always just kind of oscillate and wander around the region of the minimum. But it won't ever just head to the minimum and stay there. 

In practice, the mini-batch size you use will be somewhere in between.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/COMPARE.png)

- If small training set : use batch gradient descent (m < 2000)

- Typical mini batch size : (better to put power of 2)

- Make sure mini batch to fit in CPU/ GPU memory

## Exponentially weighted Moving Averages

To see few optimization algorithms, Which are  faster than gradient descent. In order to understand those algorithms, you need to be able they use something called exponentially weighted averages.

**Vt = (beta) V(t-1) + (1-beta) theta(t)**  ( Vt averaging over 1/(1-B)  = days)

1. With higher value of B the plot we get its smoother

2. With higher value of B we are averaging in bigger window

3. With higher value of B we are giving higher weight to previous days and smaller value to today.

4. With higher value of B this exponentially weighted averages adapt slower to changes.

To see on how many previous days we need to focus we chech (beta) ^ x = 1/e and x is the day we need to focus on previous days.

## Bias correction

To modify this estimate that makes it much better and makes it more accurate, especially during this **initial phase of your estimate** its better to : **Vt / (1 - Bt)**

Bias correction can help to have a better estimate in early phases.

## Gradient Descent with Momentum

This type of gradient descent almost always works faster than standard gradient descent algorithm.

**The basic idea is to compute an exponentially weighted average of your gradients, and then use that gradient to update your weights instead.**

In gradient descent we might have some noises of horizontal lines (makes gradient descent slow) and on horizontal if we have a big learning rate we might get shotted.

So we want to be faster in horizontal and slower in vertical axis.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/momentum.png)

## RMSprop

There's another algorithm called RMSprop, Which stands for **root mean square prop**, that can also speed up gradient descent.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/rmsprop.png)

Since w are in horizontal axis, we hope sdw be small which deviding by a small number make us to move larger step in horizontal axis.

Since b is vertical axis, we hope sdp be big which deviding by a big value makes us to move smaller step in vertical axis.

## Adam optimization algorithm

Adaptive moment estimation, Adam optimizer is combining RMSprop and momentum together:

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/adam.png)

## Learning Rate Decay

One of the things that might help speed up your learning algorithm is to slowly **reduce your learning rate over time.** We call this learning rate decay.

If you were to slowly reduce your learning rate Alpha, then during the initial phases, while your learning rate Alpha is still large, you can still have relatively fast learning. But then as Alpha gets smaller, your steps you take will be slower and smaller, and so, you end up oscillating in a tighter region around this minimum rather than wandering far away even as training goes on and on.

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/learning-rate-decay.png" title="" alt="" data-align="center">

Try different alpha zero and decay rate.

- Other learning rate decay methods
  
  ![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/other-decay.png)

- If our model takes days to train, some people do manual decay. 

## Local optima

if you create a neural network, most points of zero gradients are not local optima, Instead most points of zero gradient in a cost function are **saddle points**.

If local optima aren't a problem, then what is a problem? 

It turns out that plateaus can really slow down learning and a plateau is a region where the derivative is close to zero for a long time.

1. we're actually pretty unlikely to get stuck in bad local optima so long as you're training a reasonably large neural network, save a lot of parameters, and the cost function J is defined over a relatively high dimensional space. 

2. But second, that plateaus are a problem and you can actually make learning pretty slow. And this is where algorithms like momentum or RmsProp or Adam can really help your learning algorithm as well. 

And these are scenarios where more sophisticated observation algorithms, such as Adam, can actually speed up the rate at which you could move down the plateau and then get off the plateau. So because your network is solving optimizations problems over such high dimensional spaces, to be honest, I don't think anyone has great intuitions about what these spaces really look like, and our understanding of them is still evolving. 


