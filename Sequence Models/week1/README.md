# Week 1

We cant use standard network because : 

1. Inputs, outputs can be differnet lengths in different examples.

2. Doesnt share features learned across different positions of text.

## Recurrent neural networks

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/RNN.png)

One of the downside of recurrent neural netowrk is in this particular neural network structure is that the prediction at a certain time uses inputs or uses information from the inputs earlier in the sequence but not information later in the sequence. (Solution: bidirection RNN).

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/ff-RNN.png)

Compress Waa and Wax into one Wa.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/sff-RNN.png)

And we use backpropagation in RNN as well.

In RNN the input and output text have same length but in real world we have different input and output text size.

## Different types of RNN

1. many to many
   
   input with T length, output with T length 
   
   <img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/many-to-many.png" title="" alt="" width="237">
   
   they can have different length as well but must be more than 1
   
   <img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/many-to-many2.png" title="" alt="" width="249">

2. many to one
   
   input with T length, output with 1 length
   
   example: sentiment analysis
   
   ![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/many-to-one.png)

3. One to many
   
   Input with length 1, output with T length
   
   example : music generation
   
   ![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/one-to-many.png)

4. One to one
   
   ![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/one-to-one.png)

## Language model and sequence generation

The task of langugae model is to tell probability of a sentence (sequence of words).

Training set : large corpus of english text.

- Tokenize the sentence

- Map each word to one-hot vector.

- Add EOS token for end of sentence.

- Replace words that are not in our vocab with token

- Then we input our tokenized sentence to RNN model.

- Y(1) calculate the probability of first word of sentence.

- Y(2) calculate the probability of second word giving it the correct first word.

- Y(T) calculate the probablity of T word given first (T-1) correct words.

## Sampling a sequence token

This method is used to generate a randomly sentece using our trained language model RNN.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/sampling-training.png)

We can stop sampling by two methods:

1. When we have generated EOS token

2. Put threshold for number of sampling

## Character level language model

1. Now our vocabulary is characters. 

So many english sentences will have 10 to 20 words but may have many, many dozens of characters. And so character language models are not as good as word level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence. And character level models are also just more computationally expensive to train. So the trend I've been seeing in natural language processing is that for the most part, word level language model are still used, but as computers gets faster there are more and more applications where people are, at least in some special cases, starting to look at more character level models.

## Vanishing Gradients with RNN

We said that if this is a very deep neural network, then the gradient from this output y would have a very hard time propagating back to affect the weights of these earlier layers, to affect the computations of the earlier layers. 

For an RNN with a similar problem, you have forward prop going from left to right and then backprop going from right to left. It can be quite difficult because of the same vanishing gradients problem for the outputs of the errors associated with the later timesteps to affect the computations that are earlier.

## Exploding Gradients with RNn

Exploding gradients happens it can be catastrophic because the exponentially large gradients can cause your parameters to become so large that your neural network parameters get really messed up. It turns out that exploding gradients are easier to spot because the parameter has just blow up. You might often see NaNs, not a numbers, meaning results of a numerical overflow in your neural network computation. If you do see exploding gradients, one solution to that is apply gradients clipping. (look at your gradient vectors, and if it is bigger than some threshold, re-scale some of your gradient vectors so that it's not too big, so that is clipped according to some maximum value.)

## GRU

<img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/RNN-unit.png" title="" alt="" width="288">

There is a memory cell and gamma (which says till when should we memorize the memory cell and update the value)

If gamma = 1 means update our memory cell and put the new value in it and memorize it.

If gamma = 0 means dont update our memory cell and use previous value.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/GRUU.png)

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/GRU-unitt.png)

## LSTM

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/LSTM.png)

With both LSTM and GRU we can capture more range of dependencies. People more choose LSTM although GRU is more simple and it might be easier to scale them with bigger problems.

## Bidirectional RNN

Each of block can be RNN, LSTM or GRU block.

<img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/BRNN.png" title="" alt="" width="418">

## Deep RNNs

<img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/Deep-RNN.png" title="" alt="" width="439">
