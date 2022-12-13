# Week 4

## Face recognition vs Face Verification

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/fr-fv.png)

## One shot learning

Learning from one example to recognize the person again.

We have to have a similarity function to calculate the similarity degree of difference between images.

For having this function we use Siamese network.

## Siamese network

Input pictures to same neural networks and the output is encoding for each picture.

Compare each picture encoding (output of nn) together to find similarity or difference.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/s-n.png)

## Triplet Loss

One way to learn the parameters of the neural network, so that it gives you a good encoding for your pictures of faces, is to define and apply gradient descent on the triplet loss function.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/tl-1.png)

for choosing ancher, positive and negative if we choose them randomly they are easily satisfied, we need to choose triplets that're hard to train on.

There is another way to find similarity of images using binary classification.

## Face verification using binary classification

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/binary-classification.png)

## Neural Style transfer

<img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/neural-style.png" title="" alt="" width="342">

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/ns-cost.png)

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/style.png)

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Convolutional%20Neural%20Networks/pictures/style-cost.png)
