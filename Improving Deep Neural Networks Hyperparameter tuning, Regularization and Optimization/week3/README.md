# Week 3

## Tuning process

One of the painful things about training deepness is the sheer number of hyperparameters you have to deal with, ranging from the learning rate alpha to the momentum term beta,  or the hyperparameters for the Adam Optimization Algorithm which are beta one, beta two, and epsilon. Maybe you have to pick the number of layers, maybe you have to pick the number of hidden units for the different layers, and maybe you want to use learning rate decay, so you don't just use a single learning rate alpha. And then of course, you might need to choose the mini-batch size.

The imortances of hyperparameter:

1. Learning rate

2. Momentum term beta, number of hidden units, mini-batch size

3. Number of layers, Learning rate decay

Now, if you're trying to tune some set of hyperparameters, how do you select a set of values to explore?

**Try random values, Don't use a grid.**

**Coarse to fine** :  final scheme what you might do is zoom in to a smaller region of the hyperparameters, and then sample more density within this space.

## Using and Appropiate scale to pick hyperparameters

Sampling at random doesn't mean sampling uniformly at random, over the range of valid values. Instead, it's important to pick the appropriate scale on which to explore the hyperparameters.

**For learning rate**

1. Take log of start of our period. =a

2. Take log of end of our period. =b

3. choose r randomly from [a,b]

4. learning rate = 10 ^ r

**For weighted averages**

1. We change our period to 1-B

2. Take log of strat and end

3. Choose r randomly between [a,b]

4. 1 - B = 10 ^ r

5. B = 10 ^ r + 1

So what this whole sampling process does, is it causes you to sample more densely in the region of when beta is close to 1.

### Hyperparameter search process (panda vs. caviar)

1. **Babysitting one model**

Babysit one model, that is watching performance and patiently nudging the learning rate up or down. But that's usually what happens if you don't have enough computational capacity to train a lot of models at the same time.

2. **Training many models in parallel**

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/pandas-vs-caviar.png)

In depends on how many computational resources we have.

## Normalizing Activations in a Network

What if we want to normalize not only input of neural network but also input of hidden layers and activation functions?

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/activate-normalization.png" title="" alt="" width="546">

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/if.png)

Remeber by choosing other values of gamma and beta than above, this allows you to make the hidden unit values have other means and variances as well.

Using gamma and beta we can make sure our z(i) have the values we want and not only linear.

- Normalizing the input features, the X's, to mean zero and variance one which makes speed up.

## Fitting Batch norm into a neural network

In neural network now we have 4 parameters for each layer :  1.W, 2.b, 3.Gamma 4.betha (different from b in rmsprop,adam and momentum).

we have to optimize this hyperparameter we can use different optimization algorithm such as gradient descent

- Batch Norm is usually applied with mini-batches of your training set.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/batch-Norm.png)

## Why does batch norm works

For example in neural network with 3 hidden layer, hidden layer 3 gets input of, layer 2, so its important that what is the output of hidden layer 2 and is it normalized or not? because if differnet layers dont have same distribution it makes our weight of layers dont fit well and we will reach to covariance shift problem.

**First effect:**

1. Batch norm it reduces the amount that the distribution of these hidden unit values shifts around.

2. it limits the amount to which updating the parameters in the earlier layers can affect the distribution of values that the third layer now sees and therefore has to learn on.

3. batch norm reduces the problem of the input values changing, it really causes these values to become more stable, so that the later layers of the neural network has more firm ground to stand on.

4. even though the input distribution changes a bit, it changes less, and what this does is, even as the earlier layers keep learning, the amounts that this forces the later layers to adapt to as early as layer changes is reduced or, if you will, it weakens the coupling between what the early layers parameters has to do and what the later layers parameters have to do.

5. it allows each layer of the network to learn by itself, a little bit more independently of other layers, and this has the effect of speeding up of learning in the whole network.

6. from the perspective of one of the later layers of the neural network, the earlier layers don't get to shift around as much, because they're constrained to have the same mean and variance. And so this makes the <u>**job of learning on the later layers easier**</u>.

**Second effect:**

1. Each mini-batch is scaled by the mean/variance computed on just that mini-batch.

2. This adds <u>some noise</u> to the values z[L] within that minibatch. So similar to dropout, it adds some noise to each hidden layers activations.

3. This has a slight regularization effect.

## Batch norm at test time

In test time you might not have a mini batch of examples to process at the same time. So, you need some different way of coming up with mu and sigma squared.

During **training time** mu and sigma squared are computed on an entire mini batch of say 64 engine, 28 or some number of examples. 

But that **test time**, you might need to process a <u>single example </u>at a time. So, the way to do that is to estimate mu and sigma squared from your training set and there are many ways to do that. You could in theory run your whole training set through your final network to get mu and sigma squared. 

But *in practice*, what people usually do is implement and exponentially weighted average where you just keep track of the mu and sigma squared values you're seeing during training and use and exponentially the weighted average, also sometimes called the running average, to just get a rough estimate of mu and sigma squared and then you use those values of mu and sigma squared that test time to do the scale and you need the head and unit values Z.

## Softmax Regression

Its a generalization of logistic regression called Softmax regression. The less you make predictions where you're trying to recognize one of C or one of multiple classes, rather than just recognize two classes.

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/softmax.png" alt="" width="249" data-align="center">

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/lossfunc.png)

we want to make loss function small so we have to make the y parameter as big as possible.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/pictures/costfunct.png)
