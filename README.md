# THE DIGIT CLASSIFIER
---
## INTRODUCTION:
The Digit Classifier is a program which classifies a digit from 0 to 9 given its image. The project was built in three stages. The first stage used a Single Layer Neural Network, the second one used a Multi Layer Neural Network and the third one used a Convolutional Neural Network. The project is programmed in Python using [Numpy](https://www.numpy.org/), [Pandas](https://pandas.pydata.org/) and [Pytorch](https://pytorch.org/) libraries. The project was trained using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
***
## MACHINE LEARNING:
Machine learning is an application of artificial intelligence (AI) which provides system the ability to learn and improve without being explicitly programmed. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly. 
***
## PRE-REQUISITES FOR THE PROJECT:
1. **Python**
2. **Numpy**
3. **Pytorch**
4. **Deep Learning and Neural Networks**
5. **Convolutional Neural Networks**
***
## APPROACHES EMPLOYED:
1. **Single Layer Perceptron:-** The first approach used just a single layer of 10 neurons that took all the pixels as individual inputs and gave a prediction using a simple Soft Max Activation function. The accuracy achieved was 91.68 % on the MNIST test set. This was built from scratch using numpy.
2. **Multi Layer Perceptron:-** The second approach used a Neural Network with one hidden layer having 120 nodes.The hidden layer uses a ReLU Activation Function.The hidden layer was followed by an output layer which consists of 10 nodes using Soft Max Activation function. The accuracy achieved was 97.03 % on the MNIST test set. This was built from scratch using numpy.
3. **Convolutional Neural Network:-** The last approach used a Convolutional Neural Network using 2 Convolution layers and 2 Fully connected layers. The convolution layers first apply convolution using a 5x5 kernel and then apply Max Pooling. The fully connected layers first use a ReLU activation followed by a Soft Max Activation in the output layer. The accuracy achieved was 97.73 % on the MNIST test set. This was built using Pytorch.
***
## SOURCES USED:
1. The [Deep Learning and Neural Networks](https://www.coursera.org/learn/neural-networks-deep-learning) and [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?=) courses from [coursera](https://www.coursera.org/) helped us in getting acquanted with the basic concepts of Deep Learning and Convolutional Neural Networks. 
2. [StackOverflow](https://stackoverflow.com/) was continuously checked in case of errors encountered during coding of the project.
***
## MENTORS:
1. **Khush Agrawal**
2. **Himanshu Patil**
3. **Rohit Lal**

Special thanks to all the other mentors and members of [IV LABS](http://www.ivlabs.in/), without their support this project wouldn’t have been completed successfully.
***
