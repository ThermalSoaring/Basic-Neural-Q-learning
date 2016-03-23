# David Egolf
# For senior project
# March 2016
# Based on example td.py from the PyBrain library

import sys # Allows us to force printing
import simpleThermal as simpTh # Contains easy to use Gaussian function
import simpleThermalEnvironment as thermEnv # Very basic distance updating environment

import numpy as np
from math import pi

# Plotting libraries:
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable # For temporary lookup table approach to getting values for action in states

# Plots decision regions:
# There are numDist chunks, so we need numDist - 1 lines to separate these chunks
# If we had 4 chunks, we have the one catch all and then 3 closer chunks, the first at outBound/3 = outBound/(numDist - 1)
# The last is at 2*outBound/(numDist - 1). We also have a line at outBound.    
def drawChunks(outBound, numDist):
    plt.axhline(y=outBound,linewidth=1, color='r')    
    firstLine = outBound/(numDist - 1)
    for i in range(numDist - 2):
        plt.axhline(y=firstLine*(i+1),linewidth=1, color='r')  


# Create environment
''' 
Environment: The general interface for whatever we would like to model, learn about,
predict, or simply interact in. We can perform actions, and access
(partial) observations.
'''
maxPlaneStartDist = 8
numAngs = 6 # Discretizing allowed turning directions into this many chunks
numDist = 6 # Discretizing distances from center into this many chunks
thermRadius = 3; # Standard deviation of reward function 
stepSize = 0.1 #maxPlaneStartDist/(numDist-1) # Will oscillate about maximum if following a good policy
env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)  

# Print starting distance 
print('Start distance:')
print(env.distPlane())
  
# Define an learning agent
# A learning agent is an agent:
'''
An agent is an entity capable of producing actions, based on previous observations.
Generally it will also learn from experience. It can interact directly with a Task.
'''

# More specifically:
''' LearningAgent has a module, a learner, that modifies the module, and an explorer,
which perturbs the actions.
'''

# Define a learner for the learning agent
from pybrain.rl.learners import SARSA
learner = SARSA()

# Create a module for the learning agent
# Beginning with a table - eventually want a neural network here
table = ActionValueTable(numDist, numAngs,'panda') # number of states (10 discretized distances), number of actions (travel towards, away or at right angles to thermal)
table.initialize(0.5) # Set initial positive value to all actions in all states (encourages exploration)

# For each state, prints out the values of the different actions
for i in range(numDist):
    print(table.getActionValues(i))

   
# Create a learning agent, using both the module and the learner
from learnerCustom import CustLearningAgent
#agent = LearningAgent(table, learner)
agent = CustLearningAgent(table, learner)
agent.name = 'ocelot'

# Add an explorer to the learner (use default values - this code added for clarity)
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
eps = 0.3
epsDecay = 0.9999
agent.explorer = EpsilonGreedyExplorer(eps,epsDecay)

# We now to need to make an experiment
# An agent requries both a task and a learning agent
from pybrain.rl.experiments import Experiment
'''
An experiment matches up a task with an agent and handles their interactions.    
'''
# We need to define a task:
'''
A task is associating a purpose with an environment. It decides how to evaluate the
observations, potentially returning reinforcement rewards or fitness values.
Furthermore it is a filter for what should be visible to the agent.
Also, it can potentially act as a filter on how actions are transmitted to the environment.
'''
# We construct a task, using the defined environment
from simpleThermalTask import SimpThermTask
task = SimpThermTask(env)

# We now have all the pieces to make our experiment
experiment = Experiment(task, agent)

print('\n\n Begin learning.\n\n')

# Learn!

# Reset the plane to its starting point trainEpochs times
trainEpochs = 60
for j in range(trainEpochs):
    # Reset the experiment, keeping the learned information
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)
    task = SimpThermTask(env)
    experiment = Experiment(task, agent)

    # Repeat the interaction - learn cycle several times
    numTrain = 30;
    numInterPerTrain = 30;
    for i in range(numTrain):
        '''print('Position of plane:')
        print(env.distPlane())
        '''#print('Position index:', env.getSensors())
        
        # We interact with the environment without updating value estimates
        experiment.doInteractions(numInterPerTrain) 

        # Using the data from the interactions, update value estimates
        agent.learn()
        agent.reset()    
        
        #Print updated estimates of actions in different states
        #for i in range(numDist):
        #    print(table.getActionValues(i))
        #print('\n')
    
# Print the lookup table of state-action values
print('Learned state-action values:')
for i in range(numDist):
    print(table.getActionValues(i))
print('\n\n')
print('Hit ENTER to begin testing.\n\n')
input()

# An example table of state-action values
'''
[ 13.29015278  12.41753131  12.39088926  12.60219653]
[ 12.19686764   8.51269625   8.75278271   8.51839986]
[ 10.80527491   6.78973061   4.18381077   4.66433501]
[ 4.17227096  0.49664927  0.49982134  0.49814968]
[ 0.5  0.5  0.5  0.5]
'''

# Using the look-up table, have the plane move to the high reward area
testTable = True
if (testTable):
    # Turn off exploration
    learner._setExplorer(EpsilonGreedyExplorer(0))
    agent = CustLearningAgent(table, learner)    
    #agent = LearningAgent(table, learner)

    # Move the plane back to the start by resetting the environment
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)
    task = SimpThermTask(env)
    experiment = Experiment(task, agent)

    # Have the plane move 100 times, and plot the position of the plane (hopefully it moves to the high reward area)
    # Breakpoint code: import pdb; pdb.set_trace()    
    testIter = 100
    trainResults = [env.distPlane()]
    for i in range(testIter):
        #print('Pos:',env.distPlane()) 
        
        experiment.doInteractions(1) 
        trainResults.append(env.distPlane())
    
    import sys; sys.stdout.flush();
        
    # Plot the training results
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(trainResults,'o')
    plt.ylabel('Distance from center of thermal')
    plt.xlabel('Interaction iteration')
    plt.title('Test Results for SARSA Table Based Learner')
    drawChunks(maxPlaneStartDist*1.5, numDist) 
    
 
# Create a neural network using the lookup table
# Given a state (which may include several inputs, such as distance to thermal), its activation is a tuple that estimates the value of each action in that state
# We train it using the lookup table generated above, for the moment
## This training data is stored in "table"

from pybrain.datasets import SupervisedDataSet

# Set up form of training data
numInput = 1        # Number of input features
numOutput = numAngs # Number of actions we want values for (directions we could travel in)
ds = SupervisedDataSet(numInput, numOutput) 

# Add items to the training dataset
for i in range(numDist):
    inData = tuple([i]) # i is the index of the distance chunk we are from the estimated center of thermal (higher i = larger distance)
    outData = tuple(table.getActionValues(i))
    # Add the data to the datasets
    ds.appendLinked(inData,outData)

# Build a feed forward neural network (with a single hidden layer)
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
numHidden = 20
net = buildNetwork(ds.indim, 	# Number of input units
                   numHidden, 	# Number of hidden units
                   ds.outdim, 	# Number of output units
                   bias = True,
                   hiddenclass = SigmoidLayer,
                   outclass = LinearLayer # Allows for a large output
                   )	
                  
#----------
# Train network
#----------
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, verbose = True)
trainer.trainUntilConvergence(maxEpochs = 100)
    
print(ds)

# Print the activation of the network in the different states
# Ideally, this should be very similar to the "table" object trained using SARSA learning
for i in range(numDist):
    print(net.activate([i]))

    
testNet = True
if (testNet):
    # Turn off exploration
    learner._setExplorer(EpsilonGreedyExplorer(0))
    agent = CustLearningAgent(net, learner)
    
    # Move the plane back to the start by resetting the environment
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)
    task = SimpThermTask(env)
    experiment = Experiment(task, agent)

    # Have the plane move 100 times, and plot the position of the plane (hopefully it moves to the high reward area)
    # Breakpoint code: import pdb; pdb.set_trace()    
    testIter = 100
    trainResults = [env.distPlane()]
    for i in range(testIter):
        experiment.doInteractions(1) 
        trainResults.append(env.distPlane())       

    import sys; sys.stdout.flush(); # Force printing
        
    # Plot the training results
    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.plot(trainResults,'o')
    plt.ylabel('Distance from center of thermal')
    plt.xlabel('Interaction iteration')
    plt.title('UAV Following Neural Network Trained from SARSA Table')
    
    outBound = maxPlaneStartDist*1.5   
    drawChunks(maxPlaneStartDist*1.5, numDist)   
    plt.show()
    


