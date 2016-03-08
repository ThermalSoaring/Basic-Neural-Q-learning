# David Egolf
# For senior project
# March 2016
# Based on example td.py

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

# Create environment
''' 
Environment: The general interface for whatever we would like to model, learn about,
predict, or simply interact in. We can perform actions, and access
(partial) observations.
'''
maxPlaneStartDist = 8
numAngs = 5 # Discretizing allowed turning directions into this many chunks
numDist = 6 # Discretizing distances from center into this many chunks
thermRadius = 3; # Standard deviation of reward function 
stepSize = 0.1 #maxPlaneStartDist/(numDist-1) # Will oscillate about maximum if following a good policy
env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)  

# Print starting distance 
print('Start distance:')
print(env.distPlane())

# Test out the environment - command the plane to move towards the center or away
'''
for x in range(0, 5):
    # Move the plane towards the thermal
    # Setting theta = 0 moves it directly towards, theta = 1 moves it away
    theta = 0
    env.performAction(theta)
    
    # Print the new distance to the center
    print(env.distPlane())
'''
  
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
agent = LearningAgent(table, learner)
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
trainEpochs = 50
for j in range(trainEpochs):
    # Reset the experiment, keeping the learned information
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)
    task = SimpThermTask(env)
    experiment = Experiment(task, agent)

    # Repeat the interaction - learn cycle several times
    numTrain = 20;
    numInterPerTrain = 20;
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
    
# Test!
print('Learned state-action values:')
for i in range(numDist):
    print(table.getActionValues(i))
print('\n\n')
print('Hit ENTER to begin testing.\n\n')
input()

# Turn off exploration
learner._setExplorer(EpsilonGreedyExplorer(0))
agent = LearningAgent(table, learner)

# Move the plane back to the start by resetting the environment
env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)
task = SimpThermTask(env)
experiment = Experiment(task, agent)


# Breakpoint code: import pdb; pdb.set_trace()    

testIter = 100
trainResults = [env.distPlane()]
for i in range(testIter):
    print('Pos:',env.distPlane()) 
    
    experiment.doInteractions(1) 
    trainResults.append(env.distPlane())

import sys; sys.stdout.flush();
    
# Plot the training results
import matplotlib.pyplot as plt
plt.plot(trainResults,'o')
plt.ylabel('Distance from center of thermal')
plt.xlabel('Interaction iteration')
plt.title('Test Results for SARSA Table Based Learner')
plt.show()
 

    







