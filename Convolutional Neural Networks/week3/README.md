# Week 3

We have classifcation, classification with localization, detection algorithm.

## Object localization

For object localization next to our softmax output, we have a bounding box output (bx, by, bh, bw) .

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/new-y.png)

if there is an object is picture pc = 1 and loss function will be :

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/ly=1.png)

if there is no object in picture pc = 0 and our main goal is to see how accurate is our algorithm in object detection (pc) and loss function will be : 

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/ly=0.png)

## Landmark detection

Instead of bx, by, bh, bw we will output landmark of face such as coordination of corners of eyes. This is used for emotion detection and pose detection.

## Object detection using sliding window

1. Pick a certain window size.

2. Create a ConvNet for a  small rectangular region.

3. Input the ConvNet each time the new window on the picture.

4. You keep going until you've slid the window across every position in the image.

There's a huge disadvantage of Sliding Windows Detection, which is the <u>computational cost</u>. Because you're cropping out so many different square regions in the image and running each of them independently through a ConvNet. And if you use a very coarse stride, a very big stride, a very big step size, then that will reduce the number of windows you need to pass through the ConvNet, but that courser granularity may hurt performance. Whereas if you use a very fine granularity or a very small stride, then the huge number of all these little regions you're passing through the ConvNet means that means there is a very high computational cost.

## Convolutional Implementation of sliding window

Instead of forcing you to run n propagation on n subsets of the input image independently, Instead, it combines all n into one form of computation and shares a lot of the computation in the regions of image that are common. 

This algorithm is much more computaitonally efficient but it sill has a problem of not quite outputing the most accurate bounding boxes.

For accurate bounding boxes we use YOLO.

## Intersection over union

To see how accuracte is our predicted bounding box we use down formula:

**Size of intersection / size of union**

## NON-max suppression

Your algorithm may find multiple detections of the same objects. Rather than detecting an object just once, it might detect it multiple times.

For this problem we use non-max suppression :

non-max means that you're going to output your maximal probabilities classifications but suppress the close-by ones that are non-maximal.

1. It first looks at the probabilities associated with each of these detections.

2. Takes the largest one.

3. Discard others with high IOU with the previous selected bounding box.

## Anchor boxes

One of the problems with object detection as you have seen it so far is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects?

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/anchor-boxes.png)

## Semantic segmentation with U-Net

The goal is to draw a careful outline around the object that is detected so that you know exactly which pixels belong to the object and which pixels don't.

Semantic segmentation is a very useful algorithm for many computer vision applications where the key idea is you have to take every single pixel and label every single pixel individually with the appropriate class label. As you've seen in this video, a key step to do that is to take a small set of activations and to blow it up to a bigger set of activations. In order to do that, you have to implement something called the transpose convolution, which is important operation that is used multiple times in the unit architecture.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/transpose-convolution.png)
