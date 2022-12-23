## Week 3

## Basic Models

We have some important models to use in NLP

1. Sequence to sequence for machine translation

2. Image captioning 

In machine translation the decoder module is similar to language model (used for predicting)

Machine translation is similar to building a conditional language model. 

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/machine%20-trasnaltion.png)

We want a model that maximes the output. 

We can use **greedy search** :

Going to your machine translation model and then after having picked the first word, you then pick whatever is the second word that seems most likely, then pick the third word that seems most likely. This algorithm is called greedy search. And, what you would really like is to pick the entire sequence of words, y1, y2, up to yTy, that's there, that maximizes the joint probability of that whole thing.

But we are choosing best sentence only by probability of P(y|x) which isnt correct.

## Beam search

In greedy search we only considered one output for p( y | x), but in beam search we specify beam width and we consider B output for p(y|x). We need B copy of network.

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/Beam-search.png)

In each step we choose B phrase. If B=1 we reach to greedy search.

## Refinements to Beam search

1. Length normalization
   
   these probabilities are all numbers less than one, in fact, often they're much less than one and multiplying a lot of numbers less than one result in a tiny number, which can result in numerical under-floor, meaning that is too small for the floating point of representation in your computer to store accurately.
   
   ![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/length-normalization.png)

2. How to choose B?
   
   Large B : better result, slower
   
   Small B: worse result, faster

Unlike exact search algorithms like BFS (Breadth First Search) or DFS (Depth First Search), Beam search runs faster but is not guaranteed to find exact maximum for argmax p(y|x).

## Error analysis in Beam search

Using this process we can undrestand what fraction of errors are due to beam search vs RNN model.

## Attention model Intution

1. We read sentence part by part and we dont memorize whole sentence (Long sequence)
   
   ![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/attention-intution.png)

![](/Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/alpha.png)

<img title="" src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/alpha2.png" alt="" width="385">

## Speech Recognition

Because even the human ear doesn't process raw wave forms, but the human ear has physical structures that measures the amounts of intensity of different frequencies, there is, a common pre-processing step for audio data is to run your raw audio clip and generate a spectrogram.

**CTC cost for speech recognition**:

 <img title="" src="file:///Users/rojina/Desktop/ai/AI-courses/DeepLearningSpecialization/Sequence%20Models/pictures/CTC.png" alt="" width="464">

## Trigger word detection

We can tave traget labels, so when we heard trigger words we make the target output 1.


