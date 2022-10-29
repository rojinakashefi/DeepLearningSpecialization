# Week 1

Deep learning has already transformed traditional internet bussiness like websearch, advertising,health care where is getting really good at reading X-ray images and delivering personalized education, to precision agriculture, to even self driving cars and many others.

**Deep learning and Neural Network**

The term, Deep Learning, refers to training Neural Networks.

Neural network is that when you implement it, you need to give it just the input x and the output y for a number of examples in your training set and all these things in the middle, they will figure out by itself.

Input feature is connected to **every one** of these circles in the middle.

Neural netowrks are remarkably good at figuring out functions that accurately map from x to y **(supervised learning )** given enough data about x and y and enough training examples with both x and y.

**RELU Function**

A function which goes to zero sometimes and then it'll take of as a straight line. This function is called a Relu function which stands for <u>Rectified linear units</u>.

**Neural Network using Supervised Leanring**

| Input(x)          | Output(y)              | Application         | NN Type                                   |
| ----------------- | ---------------------- | ------------------- | ----------------------------------------- |
| Home features     | Price                  | Real Estate         | Standard NN                               |
| Ad,user info      | click on ad? (0/1)     | Online advertising  | Standard NN                               |
| Image             | Object (1,...,1000)    | Photo tagging       | CNN (convolutional)                       |
| Audio             | Text transcript        | Speech recognition  | RNN (recurrent used for sequence of data) |
| English           | Chinese                | Machine translation | RNN                                       |
| Image, Radar info | Position of other cars | Autonomus driving   | hybrid or custom NN                       |

1. Convolutional neural network : often used for image data

2. Recurrent neural network : often used for one-dimensional sequence data that has maybe a temporal component.
   
   ![](https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week1/pictures/nn-types.png)

**Structuted Data vs Unstructured Data**

- Structured Data means basically databases of data, meaning that each of the features, has a very well defined meaning.

- Unstructured data refers to things like audio, raw audio, or images where you might want to recognize what's in the image or text. Here the features might be the pixel values in an image or the individual words in a piece of text.

it has been much harder for **computers** to make sense of unstructured data compared to structured data and **people** are just really good at interpreting unstructured data.

Thanks to deep learning and neural networks, computers are now much better at interpreting **unstructured data** as well compared to just a few years ago. And this creates opportunities for many new exciting applications that use <u>speech recognition, image recognition, natural language processing on text</u>.

It turns out that a lot of short term economic value that neural networks are creating has also been on **structured data**, such as much better <u>advertising systems, much better profit recommendations, and just a much better ability to process the giant databases</u> that many companies have to make accurate predictions from them.

**Why is Deep Learning taking off?**



![](https://github.com/rojinakashefi/Intro-to-Artificial-Intelligence/blob/main/neural%20network%20and%20deep%20learning/week1/pictures/scale.png)



- Horizontal axis: the amount of data we have for a task

- Vertical axis: the performance on involved learning algorithms such as the accuracy of our spam classifier or our ad click predictor or the accuracy of our neural net for figuring out the position of other cars for our self-driving car.

- **Red line** : Traditional learning algorithm like support vector machine or logistic regression. 
  
  As a function of the amount of data you have you might get a curve that looks like this where the performance improves for a while as you add more data but after a while the performance you know pretty much stay fixed.

- Over the last 10 years for a lot of problems we went from having a relatively small amount of data to having a fairly large amount of data and all of this was thanks to the digitization of a society where so much human activity is now in the digital because we spend so much time on the computers on websites on mobile apps and activities on digital devices creates data and thanks to the rise of inexpensive cameras built into our cell phones, accelerometers, all sorts of sensors in the Internet of Things. we just accumulate a lot more data more than traditional learning algorithms were able to effectively take advantage of.

- to hit very <u>high level of performance</u> then you need two things :
  
  1. First, **train a big enough neural network** in order to take advantage of the huge amount of data.(the size of the neural network, meaning just a new network, a lot of hidden units, a lot of parameters, a lot of connections.)
  
  2. Second, on the x axis you do **need a lot of data** (scale of the data).

- In this figure, in this regime of **smaller training sets** the relative ordering of the algorithms is actually not very well defined so if you don't have a lot of training data it is often up to your skill at hand engineering features that determines(quite possible using SVM or someone training even larger neural nets). **But in this small training set regime, the SVM could do better**.so  the left of the figure the relative ordering between gene algorithms is not that well defined and performance depends much more on your skill at engine features.

- In very large training sets, in the right we see large neural nets dominating the other approaches.

- In early days scaled data and compuatation made an ability to train very large neural networks either on a cpu or gpu. But lately algorithms innovation have a good result also. such as changing sigmoid function to relu function.

- One of the problems of using sigmoid functions and machine learning is that there are regions (the slope of the function) where the gradient is nearly zero and so learning becomes really slow, because when you implement gradient descent and gradient is zero the parameters just change very slowly. And so, learning is very slow whereas by changing the what's called the activation function the neural network to use this function called the value function of the rectified linear unit, or RELU, the gradient is equal to 1 for all positive values of input.so changing the algorithm because it allows that code to run much faster and this allows us to train bigger neural networks.

- The other reason that fast computation is important is that it turns out the **process of training your network is very intuitive**. Often, you have an idea for a neural network architecture and so you implement your idea and code. Implementing your idea then lets you run an experiment which tells you how well your neural network does and then by looking at it you go back to change the details of your new network and then you go around over and over. So when your new network takes a long time to train it just takes a long time to go around this cycle and there's a huge difference in your productivity. 

- Get a result back you know in ten minutes or maybe in a day you should just try a lot more ideas and be much more likely to discover in your network. And it works well for your application and so faster computation has really helped in terms of speeding up the rate at which you can get an experimental result back and this has really helped both practitioners of neural networks as well as researchers working and deep learning iterate much faster and improve your ideas much faster.
