# Week 3

You can form a neural network by stacking together a lot of little sigmoid units.

- A layer is quantities associated with stack of nodes.

- Neural network looks like logistic regression and repeating it multiple times.

- Input Layer: We have the input features, stacked up vertically and this is called the input layer of the neural network.

- Output Layer: The final layer node(s)  is responsible for generating the predicted value y.

- Hidden Layer: In a neural network that you train with supervised learning, the training set contains values of the inputs x as well as the target outputs y. So the term hidden layer refers to the fact that in the training set, the true values for these nodes in the middle are not observed in the training set. So in a summary it means all layers between output layer and input layer.

- Activation it refers to the values that different layers of the neural network are passing on to the subsequent layers. **a[0] = X** and **Predicted y = a[2]**

- When we count layers in neural networks, <u>we don't count the input layer</u>.

- The hidden layer and the output layers will have parameters associated with them.(w , b)

- The dimension of **w is (nodes in a layer, number of input features)** and dimension of **b is (nodes in a layer,1)**.

- In each node we have two computation first to calculate z and then computing the activation using sigmoid of z:
  
  <img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/node-compute.png" alt="" width="189" data-align="center">

- ![](https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/a-representation.png)

**Neural Network Representation**

1. Using Foor loop:
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/for-loop.png" title="" alt="" width="330">

2. Using Vectorization for a **Single training example**:
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/vectorization.png" title="" alt="" width="440">
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/a-vectorization.png" title="" alt="" width="177">
   
   **We call the w vector as W[1] and the b vector as B[1]**.

3. Using Vecotrization for **Multiple training example**:
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/multiple-vectorization.png" title="" alt="" width="204">
   
   If we have m training exmple we can use foor loop to compute thier predicted value using: **(i refer to number of training example)**
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/for-multiple.png" title="" alt="" width="163">
   
   We can compute multiple training example using the description below:
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/multiple-bigX.png" title="" alt="" width="160">
   
   where Z and A are vecotres seen below:
   
   <img title="" src="/z-a.png" alt="" width="218">
   These matrixes **horizontal** side correspond to **different training example**
   
   These matrixes **vertical** side corresponf to **differrent nodes** in the neural network.(hidden units number).

**Activation Functions**

When you build your neural network, one of the choices you get to make is what activation function to use in the hidden layers as well as at the output units of your neural network.

<u>The activation functions can be different for different layers.</u>

1. Sigmoid Function
   
   **If you have a binary classification use sigmoid function for activation function of output layer.**
   
   <img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/sigmoid.png" alt="" width="188">

2. Tanh Function
   
   Better than sigmoid function because of the centering of data around zero instead of 0.5 in sigmoid.
   
   <img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/tan.png" alt="" width="187">

One of the downsides of both the sigmoid function and the tanh function is that if **z is either very large or very small**, then the gradient of the derivative of the slope of this function becomes very small and ends up being close to zero and so this can **slow down gradient descent**.

3. Relu Function
   
   The **default choice** for activation function.
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/relu.png" title="" alt="" width="199">
   
   The derivative is one so long as z is positive and derivative or the slope is zero when z is negative.

4. Leaky Relu
   
   Instead of it being zero when z is negative, it just takes a slight slope.
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/leaky-relu.png" title="" alt="" width="197">

If you were to use linear activation functions or we can also call them identity activation functions, then the neural network is just outputting a linear function of the input. So unless you throw a non-linear item in there, then you're not computing more interesting functions even as you go deeper in the network.

There is just one place where you might use a linear activation function g(z) = z and that's if you are doing machine learning on the regression problem.

**When your predicting y is a real number we can use g(z) = z in output layer but we dont use linear activation function in other layers.**

**Derivatives of Activation Functions**

1. Sigmoid Function
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/sigmoid-derivatives.png" title="" alt="" width="166">

2. Tanh Function
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/tanh-derivatives.png" title="" alt="" width="171">

3. ReLu Function
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/relu-derivative.png" title="" alt="" width="175">

4. Leaky ReLU Function
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/leaky-relu-derivative.png" title="" alt="" width="191">

**Graident Descent for Neural Networks**

**We don't get derivatives for X since input x is for supervised learning and we are not trying to optimize x.**

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/gradient-descent-neural-network.png" title="" alt="" width="394">

Using vectorization for multiple training example:

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/gd-mlt-te.png" title="" alt="" width="351">

**Random Initialization**

For logistic regression, it was okay to initialize the weights to zero. But for a neural network of initialize the weights(W) to parameters to all zero and then applied gradient descent, it won't work.

It turns out initializing the bias terms b to 0 is actually okay, but initializing w to all 0s is a problem.

So you should use th terms below:

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/random.png" title="" alt="" width="286">

The problem of initialize to zero is that for any example you give it, you'll have that a1,1 and a1,2, will be equal and both of these hidden units are computing exactly the same function.

![](https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/equality.png)

Then computing back propagation their derivatives be aslo equal.

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week3/pictures/derivatives-equality.png" title="" alt="" width="171">
