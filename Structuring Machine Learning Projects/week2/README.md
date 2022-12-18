## Week2

**Manually** examining mistakes that your algorithm is making, can give you insights into what to do next.

Well, rather than spending a few months doing this, only to risk finding out at the end that it wasn't that helpful. Here's an error analysis procedure that can let you very quickly tell whether or not this could be worth your effort.

## Error analysis

1. Get 100 mislabeled dev set examples

2. Count up how many correct are in it

3. If it is small it means, even if the dev set say 100% wrong in the first place, if we want to imporve it to get corrects we have small correct in the dev set and it is not worth it.

4. If it is large it means, it means if we imporve it to get more corrects it will work out.

The conclusion of this process gives you an estimate of how worthwhile it might be to work on each of these different categories of errors.

## Cleaning up Incorrectly Labeled Data

Is it worth your while to go in to fix up some of these labels?

Deep learning algorithms are quite robust to **random errors** in the training set. So long as the total data set size is big enough and the actual percentage of errors is maybe not too high, we can ignore it.

Deep learning algorithms are less robust to **systematic errors** in the training set.

![](https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/Structuring%20Machine%20Learning%20Projects/pictures/error-analysis.png)

If you are correcting labels of dev set apply it also on test set to continue to have the same distribution.

### Build your First System Quickly, then Iterate

1. Set up dev/test set and metric 

2. Build initial system quickly

3. Use  Bias/Variance analysis & Error analysis to prioritize next steps.

## Training and Testing on Different Distributions

If we have two datasets for example one from webpage and one from mobile app. The dataset from webpage is bigger than the dataset from mobile app, but we know that in the end the input data of the model is much from mobile app.

so there are two options:

1. Add two datasets and shuffle them and split them between test/dev/train sets. The advantage of this work is  all of our sets are from same distrubition, but the main big disadvantage is in dev and test set there will be a little data from mobile app and most of it, it's from webpages

2. Split the mobile app to two parts, add first part to train set and split again the second part to two parts, one for dev set and one for test set, now all of our data in dev and test sets are from mobile app pictures, which we know we are getting as input.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Structuring%20Machine%20Learning%20Projects/pictures/spliting-two-datasets.png)

## Bias and Variance with Mismatched Data Distributions

If training and dev datas came from same distribution then we can say if we have variance or not. but if they are not from same distribution we cant say if the high variance problem is because of having different distribution or is really because of variance problem. so in order to tease out its better to have a training-dev set: Same distribution as training set, but not used for training.

Train and train-dev set have a same distribution.

Dev and test set have a same distribution.

We run or neural network on train set and for error analysis we use train-dev,dev and test set.

1. If we have a big gap between training and training-dev it means we have variance problem because these two set have a same distribution.

2. If we have a big gap between training-dev and dev error it means the problem came from mismactch data problem.

3. If we have a big gap between human-level and training set we have a avoidable bias problem.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Structuring%20Machine%20Learning%20Projects/pictures/train-dev-set.png)

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Structuring%20Machine%20Learning%20Projects/pictures/gaps.png)

## Addressing Data mismatch

1. Carry out manual error analysis to try to undrestand difference between training and dev/set sets.

2. Make training data more similar, or collect more data similar to dev/test sets.

3. Artificial data synthesis (create date the shape we want) if you're using artificial data synthesis, just be cautious and bear in mind whether or not you might be accidentally simulating data only from a tiny subset of the space of all possible examples.

## Transfer learning

One of the most powerful ideas in deep learning is that sometimes you can take knowledge the neural network has learned from one task and apply that knowledge to a separate task.

1. Take a neural network, remove the last layer and weight and put a random weight and bias instead of it. 

2. If you retrain all the parameters in the neural network then this **initial phase** of training on is sometimes called **pre-training** then if you are **updating all the weights** afterwards and then training on the another data sometimes that's called **fine tuning**.

3. A lot of the low level features such as detecting edges, detecting curves, detecting positive objects, Learning from another dataset sensitive to these feature can help better in other datas.



Transfer learning makes sense

a)  When you have **a lot of data** for the problem you're **transferring from** and usually relatively **less data** for the problem you're **transferring to**.

b) Task A and B have the same input x.

c)  Low level features from A could be helpful for learning B.

One case where transfer learning would not make sense, is if the opposite was true.

## Multi-task learning

If we want to find four features in a picture, we can use a shared neural network of them or train four seperate neural network. which experince shows using one neural network has better result.

If we have not labeled one feature in our label dataset, we can ignore it by summing only through the ones that have 0 or 1 values.

Multi-task learning makes sense:

1. Training on a set of tasks that could benefit from having shared lower-level features.

2. Usually: Amount of data you have for each task is quite similar. ( if you focus on any one task, for that to get a big boost for multi-task learning, the other tasks in aggregate need to have quite a lot more data than for that one task)

3. Can train a **big enough neural network** to do well on all the tasks.

## End-to-end Deep learning

There have been some data processing systems, or learning systems that require multiple stages of processing. And what end-to-end deep learning does, is it can take all those multiple stages, and replace it usually with just a single neural network.

If you don't have enough data to solve this end-to-end learning problem, but you do have enough data to solve sub-problems one and two, in practice, breaking this down to two sub-problems results in better performance than a pure end-to-end deep learning approach.

Example of end-to-end system: machine translation (English -> french)

Example of breaking to sub-problems : estimating child's age through hand picture.

## Whether to use End-to-end Deep Learning

pros:

1. let the data speak ( x->y )

2. less hand-designing of componenets needed

cons:

1. May need large amount of data

2. Excludes potentially useful hand-designed components

Key Question: Do you have sufficient data to learn a function of the complexity needed to map x to y?
