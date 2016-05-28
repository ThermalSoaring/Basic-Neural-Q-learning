# What does this document explain?
This document will show you how to install Pybrain and use it in your Python programs. It also links to some resources for neural networks and reinforcement learning. Finally, it conceptually discusses the PyBrain implementation of these machine learning methods.

# What is PyBrain? 
PyBrain is a Python library for machine learning tasks. For example, it can be used for making and training neural networks, as well as designing and running reinforcement learning algorithms.

# Why is this useful?
* Python is a very common, powerful, and cross-platform programming language. Therefore, the fact that PyBrain is a Python library makes it more appealing to use. 
* Neural networks are commonly used for many machine learning tasks. PyBrain provides a flexible and easy to use (and open source!) interface for working with them. Neural networks allow programs to carry out classification or function approximation tasks in a flexible and adaptable way. 
 * Here are some resources on neural nets: 
   * https://www.coursera.org/course/neuralnets 
   * https://www.coursera.org/learn/machine-learning (weeks 4 and 5)
* Reinforcement learning is a type of machine learning that is good for automating tasks in which the performance of the learning agent can be captured by a “reward function”. For example, reinforcement learning can be used to have a cart learn the way through a maze (a reward is given upon exiting the maze), or it could be used to balance a pendulum (a reward is given when the pendulum is close to the right position). If your project involves a learning task that can be evaluated using a reward function, you might consider using reinforcement learning.
* A resource on reinforcement learning:
 * https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html 

# How to Install Pybrain
Some links to documentation on installation: (you will need git and python installed already)  
http://pybrain.org/docs/ , https://github.com/pybrain/pybrain/wiki/installation , and http://pybrain.org/docs/quickstart/installation.html

### Copy the PyBrain files from Github
To copy the Pybrain files from Github, run this command in a terminal (on Windows I use the Git bash terminal):    
```
git clone git://github.com/pybrain/pybrain.git
```

### Run the installation file (copied from Github) to setup PyBrain 
To run the installation file, run this command:  
```
python setup.py install
```

# How to Import PyBrain Features
When using the PyBrain library, you can import the part you need at the moment. For example, if I want to build a neural network I might include the following import lines:  
```
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
```

These import statements are included at the top of your file. Later in your file, you can use these imported features. For example, we can now do this:
```
valNet = buildNetwork(dimState, # Number of input units
                   20,          		# Number of hidden units 
                   1, 	        		# Number of output units
                   bias = True,
                   hiddenclass = SigmoidLayer,
                   outclass = LinearLayer # Allows for a large output (not just between 0 and 1)
                   )	
```
Notice that we have used buildNetwork, SigmoidLayer, and LinearLayer.

# How to Create and Use a Neural Network
Here are three very helpful tutorials on how to build a neural network in PyBrain:
* http://pybrain.org/docs/quickstart/network.html (“Building a Network”)
* http://pybrain.org/docs/quickstart/dataset.html (“Building a DataSet”)
* http://pybrain.org/docs/quickstart/training.html (“Training your Network on your Dataset”)

These should help get you started. Finally, here is a brief conceptual discussion on what each of these three steps do:

## 1.	Building a Network
In this step we specify the topology of the network. We set how many input neurons, output neurons, and hidden neurons there are. PyBrain can support multiple hidden layers. 

We need one input neuron for each input feature (each piece of input information), and we need one output neuron for each output of the system (the results). The more hidden layers (and hidden neurons) present, the more powerful and nonlinear the network can be. However, adding too many hidden layers and neurons can really slow down training, and can result in models that are more complicated than necessary.

For example, if I wanted to classify handwritten digits I might have 200 input neurons (one for each pixel in the image to be classified) and 10 output neurons (one for each possible classification result - either 0,1,2,3,4,5,6,7,8 or 9). 

Each neuron is connected to neurons in one or more other layers. How strongly these connected neurons affect this neuron is determined by the “weight” of these connections. The way the neural network learns is by changing these weights to improve its output.

## 2.	Building a DataSet

In this step we create a collection of training data for the network to learn from. A SupervisedDataSet consists of pairs of objects. Each pair has a set of inputs, and the corresponding desired output. Consider the following:
```
	ds.addSample((0, 1), (1,))
```
This line of code indicates that when the first input neuron receives a 0, and the second input neuron receives a 1, then the desired output is a 1. This information is stored in the SupervisedDataSet called ds.

The larger the number of items in the training data, the better. The more training examples there are, the better the neural network can learn the desired input to output relationship.

## 3.	Training your Network on your Dataset

In this step, we use the neural network topology created in #1 and the training examples created in #2 to make the neural network learn. 

We can specify which learning algorithm PyBrain uses. One very common algorithm is called backpropogation. This method feeds the neural network the input from the current training example, and then compares the output of the network to the desired output indicated by the current training example. Generally the desired output won’t quite match the real output – there is some error.  The network weights are then updated a little in a way to make the error of the output decrease as much as possible (gradient descent).

# PyBrain and Reinforcement Learning 

To understand how PyBrain’s reinforcement learning framework works, you have to spend some time crawling through the source code to see what relates to what. Here is a really good picture to keep in mind during this exploration: 
(Picture source: http://pybrain.org/docs/tutorial/reinforcement-learning.html) 

Here is a brief conceptual example to give an idea of what each of these parts do. Again, you need to spend some time with the library files to understand what is happening.

## Experiment: 
The experiment contains everything. Once everything is set up, then you can just tell the experiment to run. For example, consider this code:
```
experiment.doInteractions(numIterPerTrain)
```
This code will cause the agent to interact with the environment numIterPerTrain times. The states visited by the agent, the actions chosen by the agent, and the reward given to the agent will be saved.

To actually have the agent learn, do something like this, after calling doInteractions():
```
	agent.learn()        
    agent.reset()
```
The first line makes the agent learn based on the experiences it just went through (states, actions, and rewards). The second line (I believe) deletes the information about the experience it just went through, preparing it for the next round of interactions with the environment.

## Agent:
The agent is the thing that interacts with the environment, and that you want to learn how to interact with the environment in a way that maximizes reward. For example, for our thermal soaring project the UAV was our agent. The agent selects actions (ex. start orbiting), and receives rewards and observations (how good was what I did, and where am I now). The agent also learns from its experience, using something like Q learning to update its estimates of how good different actions are in different states. 

## Environment:
The environment keeps track of the physics of the system. It keeps track of the state of the agent, and can update it in response to the actions of the agent. For example, our environment stored the height and position of the UAV, and could update the height of the UAV based on thermal strength at the current position of the UAV.

## Task:
The task acts as a barrier between the agent and the environment. It can choose what information is provided to the learning agent by the environment, and it can choose how to filter or modify actions provided by the learning agent before it provides those to the environment.
