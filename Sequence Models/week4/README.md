# Week 4

## Transformer Network Motivation

As we move from RNN to GRU and then LSTM the models become more complex and also they are still sequential.

- Attention + CNN
  
  Self-Attention
  
  Multi-Head Attention

## Self-Attention

A (q, k, V) = attention-based vector representation of a word.

(calculate **for each word**)

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/self-att.png)

Q is the question we ask from the word.

We multiply Q with all of the keys we have to see how good each key is for each question.

Apply softmax to see which one is higher and better.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/self-attention.png)

If we apply the above operation for all A1, A2, A3, ..., A5, we get this formula:

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/formula.png)

## Multi-Head attention

We apply the above operation K times and we take average of all outputs.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/Multi-head-attention.png)

## Transformer Network

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/transformation.png)
