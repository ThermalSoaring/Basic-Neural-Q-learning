from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer


import simpleThermal as simpTh              # Import easy to use Gaussian function
import simpleThermalEnvironment as thermEnv # Describes the thermal environment (update sensor values). Sensor values discretized.
import contThermalEnvironment as contEnv    # Same as above, but with continuous sensor values
import numpy as np
from math import pi

# Import plotting functions
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Import reinforcement learning libraries
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable     # For lookup table approach to getting state-action (SA) values
from pybrain.rl.learners.valuebased import ActionValueNetwork   # For neural network approach to getting SA values

from learnerCustom import CustLearningAgent # Allows for easy training of neural network from table
from simpleThermalTask import SimpThermTask
from pybrain.rl.experiments import Experiment

import sys # Allows us to force printing

# Create environment
''' 
Environment: The general interface for whatever we would like to model, learn about,
predict, or simply interact in. We can perform actions, and access
(partial) observations.
'''
def createEnvDisc(maxPlaneStartDist, numAngs, numDist, thermRadius, stepSize): # Returns discrete sensor values. Used for table-lookup approach.
    env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)  
    return env

# Define a learning agent
# A learning agent is an agent:
'''
An agent is an entity capable of producing actions, based on previous observations.
Generally it will also learn from experience. It can interact directly with a Task.
'''
# A learning agent has the following components:
''' LearningAgent has a module, a learner, that modifies the module, and an explorer,
which perturbs the actions.
'''
# Module examples: lookup table or neural network
# Learner examples: Q or SARSA learning
# Explorer example: epsilon greedy 

def createAgentDisc(numDist, numAngs): # Learning agent for table-lookup approach.
    # Create a learner for the learning agent
    from pybrain.rl.learners import SARSA
    learner = SARSA()
    
    # We use a lookup tabel to access SA values
    # numDist = number of discretized distances we can be from the thermal center
    # numAng = number of angles we can travel in 
    table = ActionValueTable(numDist, numAngs,'myTable') 
    table.initialize(0.5) # Set initial positive value to all actions in all states (encourages exploration)

    # For each state, prints out the values of the different actions (using TABLE)
    for i in range(numDist):
        print(table.getActionValues(i))
        
    from learnerCustom import CustLearningAgent # Allows for easy training of neural network from table
    agent = CustLearningAgent(table, learner)
    agent.name = 'tableLearningAgent'     
    
    return agent    

# Add an epsilon greedy explorer to the learner (using default values)
def addEpsGreedExplorer(agent):

    from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
    eps = 0.3
    epsDecay = 0.9999
    agent.explorer = EpsilonGreedyExplorer(eps,epsDecay)
    return agent    

# trainEpochs = the number of times the environment is reset in training
# numTrain = number of times we do numIterPerTrain interactions before resetting
# numIterPerTrain = number of times we interact before learning anything    
def learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain):

    import copy
    envBackup = copy.deepcopy(env)
    for j in range(trainEpochs):
        print('epoch: ', j); sys.stdout.flush();
        # Reset the environment, keeping the learned information (in the agent)
        env = envBackup
        experiment = createExp(envBackup, agent)

        # Repeat the interaction - learn cycle several times
        for i in range(numTrain):
            
            # We interact with the environment without updating value estimates
            experiment.doInteractions(numIterPerTrain) 
            
            # Using the data from the interactions, update value estimates in the agent's module, using the agent's learner
            agent.learn()        
            agent.reset()  
    return agent
    
# Make an experiment 
'''
An experiment matches up a task with an agent and handles their interactions.    
'''
# A task is:
'''
A task is associating a purpose with an environment. It decides how to evaluate the
observations, potentially returning reinforcement rewards or fitness values.
Furthermore it is a filter for what should be visible to the agent.
Also, it can potentially act as a filter on how actions are transmitted to the environment.
'''
def createExp(env, agent): 

    # We construct a task, using the defined environment
    from simpleThermalTask import SimpThermTask
    task = SimpThermTask(env)

    # We stick the pieces together to form our experiment
    from pybrain.rl.experiments import Experiment
    experiment = Experiment(task, agent)   
    return experiment
    
# For table based method - plots decision regions:
# There are numDist chunks, so we need numDist - 1 lines to separate these chunks
# If we had 4 chunks, we have the one catch all and then 3 closer chunks, the first at outBound/3 = outBound/(numDist - 1)
# The last is at 2*outBound/(numDist - 1). We also have a line at outBound.    
def drawChunks(outBound, numDist):
    plt.axhline(y=outBound,linewidth=1, color='r')    
    firstLine = outBound/(numDist - 1)
    for i in range(numDist - 2):
        plt.axhline(y=firstLine*(i+1),linewidth=1, color='r')  

# Print the lookup table of state-action values
def printTab(table, numDist):
    
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
        
# Consider restructuring argumenets
def testTable(learner, table, env, maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius):
    # Turn off exploration
    learner._setExplorer(EpsilonGreedyExplorer(0))
    agent = CustLearningAgent(table, learner)    

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
        
    import sys                                  # Import function that forces printing
    sys.stdout.flush();
        
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
    # Train network
    from pybrain.supervised.trainers import BackpropTrainer
    trainer = BackpropTrainer(net, ds, verbose = True)
    trainer.trainUntilConvergence(maxEpochs = 200)        
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
      
# Use a lookup table to find values of actions
def tableBasedMethod(maxPlaneStartDist,numAngs,numDist,thermRadius,stepSize):  

    # Create environment
    env = createEnvDisc(maxPlaneStartDist, numAngs, numDist, thermRadius, stepSize)
    print('Start distance:')
    print(env.distPlane())

    # Create learning agent
    dimState = 1 # Currently we only record the distance to the thermal center
    agent = createAgentDisc(numDist, numAngs)
    agent = addEpsGreedExplorer(agent)

    # Learning
    print('\n\n Begin learning.\n\n')
    sys.stdout.flush();
    
    # trainEpochs = the number of times the environment is reset in training
    # numTrain = number of times we do numIterPerTrain interactions before resetting
    # numIterPerTrain = number of times we interact before learning anything
    trainEpochs = 50; numTrain = 30; numIterPerTrain = 30
    agent = learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain)
        
    table = agent.module
    printTab(table, numDist)
    
    # Test results
    testTable(agent.learner, table, env, maxPlaneStartDist, stepSize,numAngs,numDist,thermRadius)

    