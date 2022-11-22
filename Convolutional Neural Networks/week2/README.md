# Week 2

## LeNet-5 architecture

- Goal of leNet-5 was to recognize handwritten digits.

- Was trained on grayscale images.

- At that time only sigmoid and tanh activation function was used (no ReLU)
  
  ![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/leNet-5.png)

## AlexNet

- Similar to leNet but much bigger.

- Used ReLU

- training on multiple GPUs.

- local response normalization. (for example if we have 13x13x256 we will take a point and normalize it in dimension)

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/normalization.png" title="" alt="" width="108">

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/AlexNet.png)

## VGG - 16

- As you go deeper and height and width goes down, it just goes down by a factor of two each time for the pulling layers whereas the number of channels increases.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/VGG%20-%2016.png)

## ResNets

Very, very deep neural networks are difficult to train, because of vanishing and exploding gradient types of problems.

Using skip connections which allows you to take the activation from one layer and suddenly feed it to another layer even much deeper in the neural network. And using that, you'll build ResNet which enables you to train very, very deep networks.

- **Short cut** or **skip connection** must be implemented **after linear part** but **before relu part**.

<img title="" src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/Residual-block.png" alt="" width="495" data-align="center">

For deep network we make residual blocks and stack them together.

### Plain network vs Residual network

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/PvsR.png)

**Plain network :**

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/Plain.png" title="" alt="" width="168">

**Res Net:**

<img src="https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/ResNet.png" title="" alt="" width="174">

**Because we use L2 noramlization it will shrink down W values and if W shrinks to zero we will have a[L+2] = a[L] which means the layers between them have no affect on your neural network.**

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/a[l]=a[l+2].png)

1. we assume z[l+2] has same dimension as a[l] because of that we use lots of same convolutions.

2. also we can add a **WS** matrix before a[l] to make dimensions the same, it can be a fixed matrix, our weight matrix (its used when we have pooling layers)

## Inception Network Motivation

### One-by-one convolution

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/one-one-conv.png)

- Element wise product between volumes and summing the result.

- Its used in inception networks.

- If we want to shrink hight and width we will use pooling layer.

- **If we want to shrink number of channels we use 1x1 convolutions.**

Instead of saying what filter size you want, what convolutional layer or pooling layer. The inception module lets you say let's do them all, and let's concatenate the results. and then we run to the problem of computational cost. And what you saw here was how using a 1 by 1 convolution, you can create this bottleneck layer thereby reducing the computational cost significantly.

- Helps to reduce compuational cost 10 times lower without hurting the performance.
  
  ![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/using-1x1.png)

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/inception-module.png)

**Inception network** Which is largely the inception module repeated a bunch of times 

throughout the network.

## MobileNet

Using MobileNets will allow you to build and deploy new networks that work even in low compute environment, such as a mobile phone.

- Low computational cost at deployment

- Useful for mobile and embedded vision applications

- Key idea: Normal vs depthwise separable convolutions

**FOCUS ON DIMENSIONS AND NUMBER OF FILTERS**

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/normal-convolution.png)

**In depthwise convolution each of nc filters will be applied in <u>one of the nc input</u>.**

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/depthwise-convolution.png)

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/point-wise%20conv.png)

## MobileNet Architecture

The clever idea, The cool thing about the <u>bottleneck block</u> is that it <u>enables a richer set of computations</u>, thus allow your neural network to learn richer and more complex functions, while also <u>keeping the amounts of memory</u> that is the size of the activations you need to pass from layer to layer, relatively small.

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/mobile-net.png)

![](https://github.com/rojinakashefi/DeepLearningSpecialization/blob/main/Convolutional%20Neural%20Networks/pictures/bottleneck.png)

**How can you automatically scale up or down neural networks for a particular device**?

## EfficientNet

For scaling things high and down we can have:

1. High resolution image

2. Make network deeper (vary depth or wider)

Compound scaling = simultaneosly scale up/down

If you are ever looking to adapt a neural network architecture for a particular device, look at one of the open source implementations of EfficientNet, which will help you to choose a good trade-off between r, d, and w.
