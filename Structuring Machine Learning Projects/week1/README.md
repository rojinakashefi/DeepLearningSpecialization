# Week 1

When we dont have a great accuracy after training our model, we came with some ideas such as :

1) Collect more data

2) Collect more diverse training set

3) Train algorithm longer with gradient descent

4) Try adam instead of gradient descent

5) Try bigger network

6) Try smaller network

7) Try drop out

8) Add L2 regularization

9) Change network architecture 
   
   Activation functions, hidden units, ..

One of the challenges with building machine learning systems is that there's so many things you could try, so many things you could change. Including, for example, so many hyperparameters you could tune.

but we should have the eye to know **what to tune to have what effect**. This is a process we call *orthogonalization*. we want to have have 90 degrees between the things we want to control (for example speed and angle of the car), we want to control the speed but with no affect on the angle of the car and vice versa . we want <u>one feature to just affect one thing</u>.

For machine learning projects :

1. Fit **training set** well on cost function. (bigger network, different opitimizer)

2. Fit **dev set** well on cost function. (regularization, bigger training set)

3. Fit **test set** well on cost function. (bigger dev set)

4. Performs well in read world. (change dev set or cost function)

## Single number evaluation metric

**Precision**  : of all examples our model recognizes true, what percentage are actually true?

**Recall** : of all examples are actually a cat, what percentage are correctly recognized?

There is a trade off between precision and recall. and choosing between these two are hard, so we define a new classifier.

**F1** : combining precision and recall . (harmonic mean)  

2 / (1/p) + (1/R)

Solution: 

**Having a dev set (because thats how we evaluate precision and recall ) + single number evaluation metric (F1 or Average)**

## Satisficing and Optimizing Metric

It's not always easy to combine all the things you care 

about into a single row number evaluation metric. In those cases I've found it sometimes useful to set up satisficing as well as optimizing metrics. 

For example we have two classifier accuracy and running time:

First idea is to create a linear relation between these two, but it is artifical.

Second idea is to define accuracy as optimizing (we want to maximize it as much as possible) and running time as satisficing (we want to only to statisfy one rule < 100ms we dont care about the exact time).

So if we have N metrics: choose one as optimizer (the metric that you care about the value) and N-1 to be satisficing (the metric that you care only about the threshold to be satisfied)

## Train/Dev/Test Distributions

These classifier and metrics must be calculatd on training, dev or test sets. and how we define these sets are so important.

We **shouldn't** choose dev and test from different distributions.

Choose a dev set and test set from the same distrubition to reflect data you expect to get in the future and consider important to do well on.

## Size of dev and test sets

If we have 1 million example, 98% we use for train, 1% (10,000) for dev and test.

The purpose of your test set is that, after you finish developing a system, the test set helps evaluate how good your final system is. So, the guideline is, to set your test set to big enough to give high confidence in the overall performance of your system.

For some application we don't need a high confidence in the overall performance of your final system. and thats why we only need train + dev. (not recommended) (use when we have a large dev set that we are sure we dont overfit on dev set)

## When to Change Dev/Test Sets and Metrics?

This happens when (metric + dev) prefers an algorithm which is different from preference of human. In this situation we should define a <u>new error metric</u>.

in machine learning :

1. Place the target (the metric to evaluate and tune classifiers)

2. how to do well on this target. (shoot at the target) -> change the cost function for example to get better result in defined metric.

## Compare to human-level performance

The machine learning accuracy never surpasses some theoretical limit, which is called the **Bayes optimal error**. (best possible error).

The progress of machines are quite fast until they surpass human level performance after that the progress slows down.

Till passing human level performance we can use:

1. Get labeled data from humans

2. Gain insight from manual error analysis: (why did a person get this right?)

3. Better analysis of bias/variance.

But when we pass human-level we can't use this information.

## Avoidable bias

We can compare the gap between human detection error and training/dev errors. 

if the gap is so much we should focus on reducing bias between machine and human.

and if the gap between machine and human error isnt so much we should focus on reducing variance between training and dev error.

**For some human task human-level error is a proxy for bayes error**.

Avoidable bias : call the difference between Bayes error or approximation of Bayes error and the training error.

Remeber we shouldn't try to make avoidable bias to 0% because the human error (bayes error) has also a treshold and it isn't zero. and we cant make the bayes error less than the threshold.

Having human-level error helps us to know to focus on avoidable bias techniques or variance error techniques.

![](https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/Structuring%20Machine%20Learning%20Projects/pictures/avoidable-bias.png)

## Surpassing human-level performance

Imagine we have a human-error of 0.5%  and training error of 0.3%. in this case machines are better than humans and we have surpassed human-level performance. In this case we cant get any information from human-level to know if we have  avoidable bias techniques or variance error techniques.

## Imporving your model performance

1. Fit the training set pretty well. (avoidable bias) 
   
   a. training bigger network
   
   b. train longer,better optimization algorithm (RMS prop, ADAM)
   
   c .NN architecture, hyperparameters search

2. Training set performance generalizes pretty well to dev/test set. (small variance) 
   
   a. More data
   
   b. Regularization
   
   c . drop out, data augmentation
   
   d . NN architecture, hyperparameters search

3. Look at the differnece error between training error and human-level error and between training error and dev error.


