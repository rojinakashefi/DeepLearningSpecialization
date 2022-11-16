# Week 1

## Edge detection

For **vertical edge detection** we use this matrix : 

1. Right side and left side are inverse 

2. Right side (Positive numbers )shows bright side , Left side (negative numbers) shows dark side.

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/vertical.png" title="" alt="" width="126">

For **horizontal edge detection** we use this matrix:

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/horizontal.png" alt="" width="130">

We can use different kind of filter for edge detection but **the format** (one side is positive and next side is negative) must be kept.

For example **another vertical edge detection** is

**Sobel filter:**

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/sobel.png" title="" alt="" width="193">

**Scharr filter:**

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/scharr.png" title="" alt="" width="177">

you really want to detect edges in some complicated image, maybe <u>you don't </u>

<u>need to have computer vision researchers handpick these nine numbers</u>. 

Maybe you can just **learn** them and treat the nine numbers of this matrix 

as parameters, which you can then **learn using back propagation**. 

## Padding

The size of output after convolution operator is : 

**(n - f + 1) * (n  - f + 1)**

The two downside is :

1. every time you apply a convolutional operator, your image shrinks.

2. if you look the pixel at the corner or the edge, this little pixel is touched as used only in one of the outputs, because this touches that three by three region. Whereas, if you take a pixel in the middle, say this pixel, then there are a lot of three by three regions that overlap that pixel and so, is as if pixels on the corners or on the edges are use much less in the output. So you're **throwing away a lot of the information near the edge of the image**.

Solution is to *Add padding*.

After adding padding to input picture and apple the convolution operation the output size will be :

**(n + 2p - f + 1) * (n + 2p - f + 1)**

#### Valid and Same convolution

- Valid is when we have **no padding**.

- Same is when the **output size is the same as input size** which makes p as follow.
  
  **P = ( f - 1 ) / 2**

## Strided convolution

if we have padding and stride the output dimension is :

![strided.png](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/strided.png)

## Convolutions over volume

- If we have an input with x,y,c dimensionality we have different filter size but with same dimensionality (x1,y1,c).

- Convolving 3d input with 3d filter will result in 2d output.

- Convolving 3d input with n *(3d)* filter will result in nd output.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/multiple-filters.png)

- Input is our a[0].

- Filters are our **weights**.

**Summary of notations**

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/summary.png)

As we go deeper in convolution network the **x,y will shrink** as the **dimension will increase**.

#### Type of layer in a convolutional network:

1. Convolution

2. Pooling

3. Fully connected

## Pooling layers

Its like a filter which apply a function on a specific frame.

### Max pooling

- Choose the maximum number is a frame.

- It has HYPERPARAMETERS (f,s) but no parameters to learn.

- If input has **n channels** the output will also have **n channels** after applying **max pooling**. (max pooling will be applied independently on each of those channel).

### Average pooling

- In each filter instead of getting maximum value it takes average.

<u>Usually its so rare to use padding for pooling layers.</u>

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/pooling.png)

### Convolution neural network

- Some people call (conv + pool) a layer, Some people call each one of them a layer.
- Conv layers have few parameters and most of the parameters are inside fully connected networks.

## Why convolutions?

1. Shrinks input size of FC network and have extracted features as input to FC network.

2. Parameter sharing and sparsity of connections.

3. Number of parameters to train remain small

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/parameter-sharing.png)

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/sparsity.png)
