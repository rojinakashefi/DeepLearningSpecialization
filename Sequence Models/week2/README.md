# Week 2

## Word Representation

1. one hot representation
   
   Cant undrestand relationship between words (because the inner product between any vector is zero)

2. Featurized representation : word embedding

## Visualizing word embeddings

Use t-SNE to embed high dimnesional vectors into 2D.

## Transfer learning and word embeddings

1. Learn word embeddings from large text corpus. (1-100B)

2. Transfer embedding to new task with smaller training set. (say, 100k words)

3. Optional : Continue to finetune the word embeddings with new data.

One difference between the face recognition literature and what we do in word embeddings is that, for face recognition, you wanted to train a neural network that can take as input any face picture, even a picture you've never seen before, and have a neural network compute an encoding for that new picture. Whereas what we'll do, and you'll understand this better when we go through the next few videos, whereas what we'll do for learning word embeddings is that we'll have a fixed vocabulary of, say, 10,000 words. And we'll learn a vector e1 through, say, e10,000 that just learns a fixed encoding or learns a fixed embedding for each of the words in our vocabulary.

## Embedding matirx

To get word embeddings we multiple Embedding matrix to one-hot representation of word.

## Learning word embeddings

1. Neural language model
   
   a. Input one-hot vector
   
   b. Multiply one-hot representation to Embedding matrix
   
   c. Input Embedding-matrix to a neural language model 

2. For learning word embeddings we use context words to predict target words.

## Word2vec

1. Skip-grmas
   
   a. randomly pick a word to be the context word. 
   
   b. randomly pick another word within some window to be our target word.
   
   c. can use softmax classification
   
   ![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/softmax.png)
   
   The problem is iterating through all vocab is slow.
   
   Solution:
   
   1. Hierchial softmax 
      
      Takes log |V| time.
      
      It isn't balanced perfectly.
   
   2. Negative Sampling

2. CBow
   
   Get context words to predict middle words.

## Negative Sampling

We'll pick a context word and then pick a target word and that is the first row of this table. That gives us a positive example. So context, target, and then give that a label of 1. 

And then what we'll do is for some number of times say, k times, we're going to take the same context word and then pick random words from the dictionary, whatever comes out at random from the dictionary and label all those 0, and those will be our negative examples.

We use smaller K for larger dataset and larger K for smaller dataset.

**Instead of using 1000 softmax we turn it to 1000 binary classification and we only train K of them on each iteration. The one we have choosen to be our neagative sampling.**

For selecting negative examples:

<img src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/negative-sampling.png" title="" alt="" width="265">

## GloVe

It captures Xij which means number of time j appears in context of i. (words that occur together)

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/glove.png)

For frequent words like a, the try to make F smaller and for unique words we try to make weight F bigger.

## Sentiment Classification

Is a task of looking at a piece of text and telling if someone likes or dislikes the thing.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/Sentiment-anlysis.png)

Averaging has some problems

1. Ignore word orders.
   
   Completely lacking in good taste,good service. (doesnt pay attention in lacking which makes good as negative not positive)

The solution is to use RNN:

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/S-a-RNN.png)

## Debiasing Word Embeddings

1. Identify bias direction. (The bias we want to reduce)
   
   By taking averages we can undrestand the direction.

2. Neutralize: For every word that is not definitional, project to get rid of bias.

3. Equalize pairs.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/bias.png)
