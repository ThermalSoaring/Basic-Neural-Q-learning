'''
Carries out the goals described in tableMain.py
'''
import numpy as np
from math import pi
   
# Describes the thermal environment (updates sensor values given actions). Sensor values discretized.       
import simpleThermalEnvironment as thermEnv 

# Describes the thermal environment (updates sensor values given actions). Sensor values are continuous. 
import contThermalEnvironment as contEnv    

# Import plotting functions
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Import learning assets
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable     # For lookup table approach to getting state-action (SA) values
from pybrain.rl.learners.valuebased import ActionValueNetwork   # For neural network approach to getting SA values
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer 
from pybrain.rl.experiments import Experiment
from pybrain.rl.learners import SARSA
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# Allows for easy training of neural network from table
from learnerCustom import CustLearningAgent 

# The interface between the environment and the learning agent
from simpleThermalTask import SimpThermTask

# For forcing printing
import sys 

# For creating a deep copy of the environment
import copy

''' 
Create environment

Environment: The general interface for whatever we would like to model, learn about,
predict, or simply interact in. We can perform actions, and access
(partial) observations. (PyBrain docs)

Returns discrete sensor values. Used for table-lookup approach.
'''
def createEnvDisc(maxPlaneStartDist, numAngs, numDist, thermRadius, stepSize):
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)  
    return env

''' Define a learning agent
 A learning agent is an agent:
    An agent is an entity capable of producing actions, based on previous observations.
    Generally it will also learn from experience. It can interact directly with a Task. (PyBrain docs)

 A learning agent has the following components:
    LearningAgent has a module, a learner, that modifies the module, and an explorer,
    which perturbs the actions. (PyBrain docs)

    Module examples: lookup table or neural network
    Learner examples: Q or SARSA learner
    Explorer example: epsilon greedy explorer
'''
# Print the value of each action in numDist states, according to table
def printDiscAgent(numDist, table):
    print('Discrete agent value estimates:')
    for i in range(numDist):
        print(table.getActionValues(i))
        
    '''
    An example Q-value table: (row = state, column = action)
    [ 13.29015278  12.41753131  12.39088926  12.60219653]
    [ 12.19686764   8.51269625   8.75278271   8.51839986]
    [ 10.80527491   6.78973061   4.18381077   4.66433501]
    [ 4.17227096  0.49664927  0.49982134  0.49814968]
    [ 0.5  0.5  0.5  0.5]
    '''
    
'''
Create learning agent for table-lookup approach
    numDist = number of discretized distances we can be from the thermal center
    numAng = number of angles we can travel in 
'''
def createAgentDisc(numDist, numAngs):     
    
    ## Create a module for the learning agent
    # We use a lookup table to access SA values
    # This creates a table to store values for numAngs actions in numDist states
    table = ActionValueTable(numDist, numAngs,'myTable') 
    table.initialize(0.5) # Set initial positive value to all actions in all states to encourage exploration

    # For each state, prints out the values of the different actions
    printDiscAgent(numDist, table)        
    
    ## Create a learner for the learning agent    
    learner = SARSA()
    
    ## Create the learning agent
    agent = CustLearningAgent(table, learner)
    agent.name = 'tableLearningAgent'     
    
    return agent    

# Add an epsilon greedy explorer to the learner
def addEpsGreedExplorer(agent):
    eps = 0.3           # How often to explore (1 = always explore, 0 = never explore)
    epsDecay = 0.9999   # Factor that decreases eps over learning iterations (explore less as certainty increases)
    agent.explorer = EpsilonGreedyExplorer(eps,epsDecay)
    return agent    

'''
The agent interacts with the environment and learns from these interactions

trainEpochs =       the number of times the environment is reset in training
numTrain =          number of times we do numIterPerTrain interactions before resetting
numIterPerTrain =   number of times the agent interacts before learning anything 
'''   
def learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain):
    # Create a copy of the environment, to revert to when we reset things
    envBackup = copy.deepcopy(env)
    for j in range(trainEpochs):
        print('epoch: ', j); sys.stdout.flush();
        
        # Reset the environment
        # Note that the learned information is kept (because that is stored in the agent)
        env = envBackup
        experiment = createExp(envBackup, agent)

        # Repeat the interaction-learn cycle numTrain times per epoch
        for i in range(numTrain):
            
            # We interact with the environment without updating value estimates
            experiment.doInteractions(numIterPerTrain) 
            
            # Using the data from the interactions, update value estimates in the agent's module, using the agent's learner
            agent.learn()        
            agent.reset() # Clear the interaction data  
    return agent
    
''' 
Create the experiment 
    An experiment matches up a task with an agent and handles their interactions. (PyBrain docs)

A task is:
    A task is for associating a purpose with an environment. It decides how to evaluate the
    observations, potentially returning reinforcement rewards or fitness values.
    Furthermore it is a filter for what should be visible to the agent.
    Also, it can potentially act as a filter on how actions are transmitted to the environment.
    (PyBrain docs)
'''
def createExp(env, agent): 
    task = SimpThermTask(env)
    experiment = Experiment(task, agent)   
    return experiment

'''    
For table based method - plots decision regions:
-We make the same decision everywhere in a given decision region

There are numDist regions, so we need numDist - 1 lines to separate these chunks
If we had 4 chunks, we have the long distance chunk and then 3 closer chunks, the first at outBound/3 = outBound/(numDist - 1)
The last is at 2*outBound/(numDist - 1). We also have a line at outBound.  
'''  
def drawChunks(outBound, numDist):
    plt.axhline(y=outBound,linewidth=1, color='r') 
    firstLine = outBound/(numDist - 1)
    for i in range(numDist - 2):
        plt.axhline(y=firstLine*(i+1),linewidth=1, color='r') # Draw horizontal red lines 
   
'''
Following the Q values held in table, have the agent interact with the environment.
Also train a policy network from the table, and use it to interact with the environment.

Plot both results.
'''   
def testTable(learner,table,env,maxPlaneStartDist,stepSize,numAngs,numDist,thermRadius):
    # Turn off agent exploration
    learner._setExplorer(EpsilonGreedyExplorer(0))
    agent = CustLearningAgent(table, learner)    

    # Move the plane back to the starting point by resetting the environment
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist,stepSize,numAngs,numDist,thermRadius)
    task = SimpThermTask(env)
    experiment = Experiment(task, agent)

    # Have the plane move testIter times, and plot the position of the plane (hopefully it moves to the high reward area)
    # Breakpoint code: import pdb; pdb.set_trace()    
    testIter = 100
    trainResults = [env.distPlane()]
    for i in range(testIter):
        experiment.doInteractions(1) 
        trainResults.append(env.distPlane())        

    sys.stdout.flush();
        
    # Plot the training results
    plt.figure(1)
    plt.plot(trainResults,'o')
    plt.ylabel('Distance from center of thermal')
    plt.xlabel('Interaction iteration')
    plt.title('Test Results for SARSA Table Based Learner')
    drawChunks(maxPlaneStartDist*1.5, numDist)     
 
    # Create a neural network using the lookup table
    # Given a state (which may include several inputs, such as distance to thermal), its activation is a tuple that estimates the value of each action in that state
    # We train it using the lookup table generated above, for the moment
    # -This training data is stored in "table"    

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
    numHidden = 20
    net = buildNetwork(ds.indim, 	# Number of input units
                       numHidden, 	# Number of hidden units
                       ds.outdim, 	# Number of output units
                       bias = True,
                       hiddenclass = SigmoidLayer,
                       outclass = LinearLayer # Allows for a large output
                       )	
    # Train network
    # DOESN'T SEEM TO ALWAYS WORK - NOT WORKING PROPERLY: DEBUG WORK NEEDED
    trainer = BackpropTrainer(net, ds, verbose = False)
    trainer.trainUntilConvergence(maxEpochs = 200)        

    # Print the activation of the network in the different states
    # Ideally, this should be very similar to the "table" object trained using SARSA learning
    print('Neural approximation of Q values table:') 
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

        sys.stdout.flush(); # Force printing
            
        # Plot the training results        
        plt.figure(2)
        plt.plot(trainResults,'o')
        plt.ylabel('Distance from center of thermal')
        plt.xlabel('Interaction iteration')
        plt.title('UAV Following Neural Network Trained from SARSA Table')
        
        outBound = maxPlaneStartDist*1.5   
        drawChunks(maxPlaneStartDist*1.5, numDist)   
        plt.show()
      
# Creates a lookup table of Q values, using SARSA training
def tableBasedMethod(maxPlaneStartDist,numAngs,numDist,thermRadius,stepSize):  

    # Create environment
    env = createEnvDisc(maxPlaneStartDist, numAngs, numDist, thermRadius, stepSize)
    print('Start distance from thermal center:')
    print(env.distPlane()); print()

    # Create learning agent
    dimState = 1 # Position from thermal center
    agent = createAgentDisc(numDist, numAngs)
    agent = addEpsGreedExplorer(agent)

    # Learn
    print('\n\n Begin learning.\n\n')
    sys.stdout.flush();
    
    # trainEpochs =       the number of times the environment is reset in training
    # numTrain =          number of times we do numIterPerTrain interactions before resetting
    # numIterPerTrain =   number of times the agent interacts before learning anything 
    trainEpochs = 50; numTrain = 30; numIterPerTrain = 30
    agent = learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain)
    
    # Display Q value estimates after learning
    table = agent.module
    printDiscAgent(numDist, table)    
    print('\n\n')
    print('Hit ENTER to begin testing.\n\n')
    input()
    
    # Follow the learned policy and plot what happens
    # Also interpolate with neural network, follow that policy, and compare the results
    testTable(agent.learner, table, env, maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)
    