# Week 2

When implementing a neural network, you usually want to process your entire training set without using an explicit for loop to loop over your entire training set.

#### **Logistic Regression**

- Logistic regression is an algorithm for binary classification.

- In binary classification, our goal is to learn a classifier that can input represented by this feature vector x (we're going to use nx (n) to represent the dimension of the input features x) and predict whether the corresponding label y is 1 or 0.

- A single training example is represented by a pair, (x,y) where x is an x-dimensional feature vector and y, the label, is either 0 or 1.

- Your training sets will have **m** training examples. (x1, y1)  (x2, y2) up to (xm, ym) which is your last training example.

- Make X matrices by putting all x in training set in columns. (each x has n features which make we have n rows in matrix and we have m training set which make our matrix to have m columns). 
  
  X.shape = (nx,,m)
  
  <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/X-matrix.png" title="" alt="" width="188">

- We stack our output labels in Y columns. y's should be between zero ane one.
  
  Y.shape = (1, m)
  
  <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/Y-matrix.png" title="" alt="" width="233">

- Parameters = **w** (w.shape = (nx,1) )and **b** (we put b and w <u>seperate</u>)

- For predicting y labels we use sigmoid function
  
  <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/sigmoid.png" title="" alt="" width="177">
  
  <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/sigmoid-y.png" title="" alt="" width="182">

#### **Logistic Regression Cost Function**

To train the parameters W and B we need to define a cost function.

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/we-want.png" title="" alt="" width="307">

We want the output data we predict be the same as the input label.

We define loss or error function to see how well our model is doing :

**The loss function measures the discrepancy between the prediction (?‘¦Ì?(?‘–)) and the desired output (?‘¦(?‘–))**. In other words, the loss function computes the error for a single training example.

1. In logistic regression people don't usually do this.Because when you come to learn the parameters, you find that the optimization problem and becomes <u>non convex</u>. So you end up with optimization problem, you're with multiple local optimum.
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/alse-error-func.png" title="" alt="" width="210">

2. So in logistic regression we actually define a different loss function that plays a similar role as squared error but will give us an optimization problem that is <u>convex</u>.
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/cost-func.png" title="" alt="" width="352">
   
   we want loss function be as **small** as possible.
   
   <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/small-as-possible.png" title="" alt="" width="391">
   
   The **loss function** was defined with respect to a **single training example**. 
   
   But **cost function** measures how are you doing on the **entire training set**.

The cost function is **the average of the loss function of the entire training set**. it measures how well your parameters W and B are doing on your entire training set. We are going to find the parameters ?‘¤ ?‘Ž?‘›?‘‘ ?‘ that minimize the overall cost function.

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/cost-function.png" title="" alt="" width="370">

y hat is the prediction output by your logistic regression algorithm using, you know, a particular set of parameters W and B.

logistic regression can be viewed as a very, very small neural network.

#### **Gradient Descent**

We use the gradient descent algorithm to train or to learn the parameters W on your training set.

We know cost function measures how well your parameters W and B are doing on your entire training set. **so we want to find w, b that minimize J(w,b).**

1. Initialize w and b to some initial value. (for logistic regression, almost any initialization method works. Usually you Initialize the values of 0. Random initialization also works, but people don't usually do that for logistic regression.)

2. Gradient descent starts at that initial point and then takes a step in the steepest downhill direction.

Repeatedly do that until the algorithm converges.

<img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/gradient-descent.png" alt="" width="173">

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/wandbupdate.png" title="" alt="" width="180">

1. Alpha : Is the learning rate and controls how big a step we take on each iteration.

2. Deritive : update of the change you want to make to the parameters w.
   
   The definition of a **derivative is the slope of a function at the point**.We use derivative to know what **direction** to step.
   
   (ignore b for now just to make this one dimensional plot instead of a higher dimensional plot.)
   
   1. <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/working.png" title="" alt="" width="130">
      
      Here the derivative is positive. W gets updated as w minus a learning rate times the derivative and so you end up subtracting from w.
      
      So you end up taking a step to the left and so gradient descent with, make your algorithm slowly decrease the parameter.
   
   2. <img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/negative.png" title="" alt="" width="119">
      
      This point the slope here will be negative.So this end up increasing parameter.
   
   Both left and right points eventually move towards the global minimum.
- convex :  bowl shape , one global minimim .

- non-convex : many local optimum.

#### **Computation Graph**

The computations of a neural network are organized in terms of a forward pass or a **forward propagation step, in which we compute the output of the neural network**, followed by a backward pass or **back propagation step, which we use to compute gradients or compute derivatives.** The computation graph explains why it is organized this way.

Computation graph comes in handy when there is some distinguished or some special output variable, such as J, that you want to optimize. And in the case of a logistic regression, J is of course the cost function that we're trying to minimize.

- Through a left-to-right pass, you can compute the value of J.

- In order to compute derivatives there'll be a right-to-left pass.
  
  We want to calculate dj/dw (which w is input in first layer) we can compute it using first compute:
  
  d(L)/d(L-1) x d(L-1)/d(L-2) x d(L-2)/d(L-3) x ... x d(2)/d(1 or w)
  
  In this form for computing previous derivitive we use backward propagation.

For **one training example** of logistic regression calculating derivativies be like:

<img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/lr-dervivative.png" alt="" width="348">

For **m training example** of logistic regression calculating derivatives be like:

**Average** **all derivatives of training se**t example for **each weight** 

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/each-weight.png" title="" alt="" width="330">

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/algorithm.png" title="" alt="" width="329">

dw1,dw2,db are used as accumlator during whole process and they dont have (i) sign.

In contrast dz(i) is for a single training example.

**All of the above computation** is for **one step** gradient descent:

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/one-step-gd.png" title="" alt="" width="264">

There are two weekness with the calculation we write:

1. First for loop for all training example.

2. Second for loop for all features we have.

Instead of writing for loops we use Vectorization.

**One gradient descent step** without for loop:

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/vectorization.png" title="" alt="" width="185">

For multiple iteration of gradient descent you put the code above in for loop in range of number of iterations.

#### **Extra Information**

Images on your computer are stored as three separate matrices corresponding red, green and blue color channels of the image.

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/color-images.png" title="" alt="" width="246">

If your input image is 64 pixels by 64 pixels, then you would have 3 64 by 64 matrices corresponding to the red, green and blue pixel intensity values for your images.

To turn these pixel intensity values- Into a <u>feature vector</u>, we're going to do is <u>unroll</u> all of these pixel values into <u>an</u> input feature vector x. If the image is a 64 by 64 image, the <u>total dimension</u> of this vector x will be 64 by 64 by 3 because that's the <u>total numbers</u> we have in all of these matrixes.

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week2/pictures/x-colors.png" title="" alt="" width="112">
