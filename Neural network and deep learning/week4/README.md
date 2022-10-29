# Week 4

By now, you've seen forward propagation and back propagation in the context of a **neural network, with a single hidden layer**, as well as **logistic regression**, and you've learned about **vectorization**, and when it's important to **initialize the ways randomly**.

**Deep Neural Network**

We have different kind of Neural Network:

<img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/different-nn.png" alt="" width="378">

1. **Shallow model** : Logistic Regression 

2. **Deep model** : a lot of hidden layers 

---

- Shallow versus depth is a matter of degree.

- There are functions that very deep neural networks can learn that shallower models are often unable to.

- View the number of hidden layers as another hyper parameter that you could try a variety of values of, and evaluate on all that across validation data, or on your development set.

**Deep Neural Network Notation**

- L = number of layers in the network.

- n[l] = The number of node in layer l  (n[0] = input layer and n[L] = output layer)

- a[l] = Activations in layer l (a[0] = X and a[L] = y^)

- W[l] = weights for computing the value z[l] in layer l

- b[l] = used to compute z [l]

- W[l] = ( n[ l ] , n[ l-1 ] ) = dW

- b[l] = ( n[ l ] , 1 ) = db

- dim(Z[ l ]) = dim(A[ l ]) = ( n[ l ] , m) = dZ = dA (BP)

- dim(z[ l ]) = dim(a[ l ]) = ( n [ l ] , 1)

**Forward Propagation in a Deep Network**

The general rule is :

for l = 1 to L( number of layers ) :

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/general-fp-rule.png" title="" alt="" width="244">

**Deep Neural Network Representation**

1. Face detection and recognition
   
   1. Input a picture of a face .
   
   2. Then the first layer of the neural network can be feature detector or an edge detector. 
   
   3. Grouping together pixels to form edges. It can then detect the edges and group edges together to form parts of faces and by putting together lots of edges, it can start to detect different parts of faces.
   
   4. Finally, by putting together different parts of faces, it can then try to recognize or detect different types of faces.
   
   The main intuition is finding simple things like edges and then building them up. Composing them together to detect more complex things like an eye or a nose then composing those together to find even more complex things. And this type of simple to complex hierarchical representation, or compositional representation, applies in other types of data than images and face recognition as well.

2. Speech recognition system
   
   1. input an audio clip.
   
   2. The first level of a neural network might learn to detect low level audio wave form features, such as is this tone going up or dwon or white noise or sniffing sound.
   
   3. By composing low level wave forms, you'll learn to detect basic units of sound such as phonemes.
   
   4. Then composing that together maybe learn to recognize words in the audio in order to recognize entire phrases or sentences.

**Forward and Backward propagation in Deep Neural network**

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/fp-bp-dnn.png" title="" alt="" width="275">

Forward and backward propagation one step:

<img title="" src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/nemodar.png" alt="" width="184">

One iteration of gradient descent :

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/gd-iteration.png" title="" alt="" width="435">

backward propagation specific:

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/bp.png" title="" alt="" width="223">

**Hyperparameters vs Parameters**

The hyperparameters controll parameters values.

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/parameters-vs-hyperparameters.png" title="" alt="" width="373">

**Deep learning is a very empirical process**

<img src="https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week4/pictures/emp.png" title="" alt="" width="176">
