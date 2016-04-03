# David Egolf
# Senior project 2015-2016
# Using reinforcement learning and neural networks to improve thermal soaring

import sys                                  # Import function that forces printing
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
      
# Create environment
''' 
Environment: The general interface for whatever we would like to model, learn about,
predict, or simply interact in. We can perform actions, and access
(partial) observations.
'''    
def createEnvCont(maxPlaneStartDist, stepSize,numAngs,thermRadius): # Returns continuous sensor values. Used for neural network approach.
    env = contEnv.contThermEnvironment(maxPlaneStartDist, stepSize,numAngs,thermRadius)
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

# dimAct    = the dimension of the action vector
# dimState  = the dimension of the state vector (= 1 if we are only keeping track of distance to center of thermal)
def createAgentCont(dimAct,dimState):
    from pybrain.rl.learners import NFQ
    learner = NFQ() # Use neuro-fitted Q learning (use Q learning with neural network instead of lookup table)    
    
    # Create a neural network model with dimState inputs and dimAct outputs
    # Then network itself has dimState + dimAct input and 1 output
    numHidden = 20
    moduleNet = ActionValueNetwork(dimState, dimAct, numHidden); moduleNet.name = 'moduleNet' 
        
    # Create a learning agent, using both the module and the learner
    agent = LearningAgent(moduleNet, learner)
    return agent

# Prints a sampling of state-action values, as indicated by the module net
def printSAValsNet(net, states):
    for state in states:
        print(state, ':\t', np.transpose(net.getActionValues(state)))

# Print state action values relative to some state
def printSADiff(net, states, refState, refAction):
    refVal = net.getActionValues(refState)[refAction]
    for state in states:
        print(state, ':\t', np.transpose(net.getActionValues(state)) - refVal)
        
# Add an epsilon greedy explorer to the learner (using default values)
def addEpsGreedExplorer(agent):

    from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
    eps = 0.3
    epsDecay = 0.9999
    agent.explorer = EpsilonGreedyExplorer(eps,epsDecay)
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
    
# Create optimistic expectations to encourage initial exploration
# optVal = an optimistic value for a state-action
# optLoc = the state for which we are going to be optimistic
def setInitEst(optVal, optLocs, agent,maxEpochs):   
    
    # Create training data
    from pybrain.datasets import SupervisedDataSet
    from scipy import r_
    from pybrain.utilities import one_to_n    
    
    module = agent.module
    
    # Currently we have one input (location) and two outputs (corresponding to travelling towards or away)
    supervised = SupervisedDataSet(module.network.indim, 1)
    for loc in optLocs: # Go through all locations we are going to be optimistic about               
        for currAction in range(module.numActions):      
            inp = r_[loc, one_to_n(currAction, module.numActions)]
            tgt = optVal 
            supervised.addSample(inp,tgt)
    print(supervised)
    # Train
    from pybrain.supervised.trainers import BackpropTrainer
    trainer = BackpropTrainer(module.network, dataset=supervised, learningrate=0.005, batchlearning=True, verbose=False)
    trainer.trainUntilConvergence(maxEpochs = maxEpochs)

    return agent

# This method handles the learning
    # trainEpochs = the number of times the environment is reset in training
    # numTrain = number of times we do numIterPerTrain interactions before resetting
    # numIterPerTrain = number of times we interact before learning anything    
def learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain):

    import copy
    envBackup = copy.deepcopy(env) # Make a copy of the environment to revert to
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
    
# Test net based method (how well does the plane move towards the thermal)
def testNet(learner, moduleNet, env, maxPlaneStartDist, stepSize,numAngs,thermRadius):
    # Turn off exploration
    from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
    learner._setExplorer(EpsilonGreedyExplorer(0))
    agent = LearningAgent(moduleNet, learner)      

    # Move the plane back to the start by resetting the environment
    env = contEnv.contThermEnvironment(maxPlaneStartDist, stepSize,numAngs,thermRadius) 
    from simpleThermalTask import SimpThermTask
    task = SimpThermTask(env)
    from pybrain.rl.experiments import Experiment
    experiment = Experiment(task, agent)

    # Have the plane move 100 times, and plot the position of the plane (hopefully it moves to the high reward area)
    testIter = 100
    trainResults = [env.distPlane()]
    for i in range(testIter):
        experiment.doInteractions(1) 
        trainResults.append(env.distPlane())  
        
    # Plot the training results
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(trainResults,'o')
    plt.ylabel('Distance from center of thermal')
    plt.xlabel('Interaction iteration')
    plt.title('Test Results for Neural Fitted Q Learner')
    plt.show()        
    
# Use a neural network to find values of actions 
def netBasedMethod(maxPlaneStartDist,numAngs,thermRadius,stepSize):

    # Create environment
    env = createEnvCont(maxPlaneStartDist, stepSize,numAngs,thermRadius)
    print('Start distance:')
    print(env.distPlane())

    # Create learning agent
    dimState = 1 # Currently we only record the distance to the thermal center
    agent = createAgentCont(numAngs, dimState)
    agent = addEpsGreedExplorer(agent)

    # Set optimistic initial values
    optVal = 150
    optLocs = range(20)
    maxEpochs = 30
    agent = setInitEst(optVal, optLocs, agent,maxEpochs)   
    
    print('Sample initial (optimistic) SA values:')
    printSAValsNet(agent.module,range(20)) 
    #refState = 0; refAction = 0;
    #printSADiff(agent.module, range(20), refState, refAction)
    
    # Learning
    print('\n\n Begin learning.\n\n')
    sys.stdout.flush();    
    
    # trainEpochs = the number of times the environment is reset in training
    # numTrain = number of times we do numIterPerTrain interactions before resetting
    # numIterPerTrain = number of times we interact before learning anything
    trainEpochs = 30; numTrain = 1; numIterPerTrain = 5
    agent = learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain)
    
    print('Sample final SA values:')
    printSAValsNet(agent.module,range(20))  
    sys.stdout.flush();
    
    # Testing
    testNet(agent.learner, agent.module,env, maxPlaneStartDist, stepSize,numAngs,thermRadius)

# Use a lookup table to store state action vvalues
def tableMain():    
    maxPlaneStartDist = 8   # Starting plane distance from thermal, or the maximum such distance if random placement of plane is allowed
    numAngs = 5             # The number of directions in which the plane is allowed to move
    numDist = 6             # Discretizing distances from center into this many chunks
    thermRadius = 3;        # Standard deviation of thermal
    stepSize = 0.1          # How far the plane moves on each interaction
    
    # I stuck table related stuff in here 
    # Some code is duplicated from main.py, but I want to be able to work on net based stuff without worrying if the table based stuff is being messed up
    # Basically, tabledBasedMethods.py is where I'm dumping the working table based method
    import tabledBasedMethods as tbm
    tbm.tableBasedMethod(maxPlaneStartDist,numAngs,numDist,thermRadius,stepSize)

# Use a neural network to store state action values
def netMain():
    maxPlaneStartDist = 1  # Starting plane distance from thermal, or the maximum such distance if random placement of plane is allowed
    numAngs = 3             # The number of directions in which the plane is allowed to move
    thermRadius = 3;        # Standard deviation of thermal
    stepSize = 0.3          # How far the plane moves on each interaction
    netBasedMethod(maxPlaneStartDist,numAngs,thermRadius,stepSize)

tableMain() # Works reasonably well
#netMain()


