'''
Use neural fitted Q to learn how to move towards the thermal center.
Currently doesn't converge to desired answer! More work is needed. 
-Tends to think thermal center is bad!
-Learning process seems to depend heavily on initial conditions
-I suspect the neural network stuff is likely the culprit
-Some attempted fixes are in myNFQ.py
'''
import numpy as np
from math import pi

import copy                                 # To create deep copies of objects
import sys                                  # For forcing printing
import simpleThermalEnvironment as thermEnv # Describes the thermal environment (update sensor values). Sensor values discretized.
import contThermalEnvironment as contEnv    # Describes the thermal environment (update sensor values). Sensor values not discretized.

# Import plotting modules
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Import learning modules
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable     # For lookup table approach to getting state-action (SA) values
from pybrain.rl.learners.valuebased import ActionValueNetwork   # For neural network approach to getting SA values
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer

# Modified version of PyBrain's nfq
from myNFQ import NFQ
# Instead of: # from pybrain.rl.learners import NFQ

'''   
Create environment

Environment: The general interface for whatever we would like to model, learn about,
predict, or simply interact in. We can perform actions, and access
(partial) observations. (PyBrain docs)

Returns continuous sensor values. Used for neural network approach.
'''
def createEnvCont(maxPlaneStartDist, stepSize,numAngs,thermRadius): 
    # Note that contEnv returns continuous sensor values
    env = contEnv.contThermEnvironment(maxPlaneStartDist, stepSize,numAngs,thermRadius)
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
def createAgentCont(dimAct,dimState,numBatchToKeep, numIterPerTrain):
    
    # Create a learner - use neural fitted Q learning (instead of SARSA) 
    sizeBatch = numIterPerTrain
    learner = NFQ(sizeBatch, numBatchToKeep)     
    
    # Create a neural network model with dimState inputs and dimAct outputs
    # Then network itself has dimState + dimAct inputs and 1 output
    # You input a state and an action, and the neural network estimates the value of that combination
    numHidden = 20
    print('Using this many hidden layer neurons: ', numHidden)
    moduleNet = ActionValueNetwork(dimState, dimAct, numHidden); moduleNet.name = 'moduleNet' 
        
    # Create a learning agent, using both the module and the learner
    agent = LearningAgent(moduleNet, learner)
    return agent

# Prints a sample of state-action values (provided by the neural network)
# The value of each action for each state in "states" is printed
def printSAValsNet(net, states):
    for state in states:
        print(state, ':\t', np.transpose(net.getActionValues(state)))

# Print value (relative to some reference state-action pair)
def printSADiff(net, states, refState, refAction):
    refVal = net.getActionValues(refState)[refAction]
    for state in states:
        print(state, ':\t', np.transpose(net.getActionValues(state)) - refVal)
        
# Add an epsilon greedy explorer to the learner 
def addEpsGreedExplorer(agent):    
    eps = 0.3           # How often to explore (1 = always explore, 0 = never explore)
    epsDecay = 0.9999   # Factor that decreases eps over learning iterations (explore less as certainty increases)
    agent.explorer = EpsilonGreedyExplorer(eps,epsDecay)
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

    # We construct a task, using the defined environment
    from simpleThermalTask import SimpThermTask
    task = SimpThermTask(env)

    # We stick the pieces together to form our experiment
    from pybrain.rl.experiments import Experiment
    experiment = Experiment(task, agent)   
    return experiment

'''    
Create optimistic expectations to encourage initial exploration
optVal =    an optimistic value for a state-action
optLocs =   the states for which we are going to be optimistic
'''
def setInitEst(optVal, optLocs, agent,maxEpochs):   
    
    # Create training data
    from pybrain.datasets import SupervisedDataSet
    from scipy import r_
    from pybrain.utilities import one_to_n    
    
    module = agent.module
    
    # Currently we have one input (location) and two outputs (corresponding to choosing to travel towards or away)
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

'''
 This method handles the learning
trainEpochs =       the number of times the environment is reset in training
numTrain =          number of times we do numIterPerTrain interactions before resetting
numIterPerTrain =   number of times the agent interacts before learning anything    
'''
def learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain):    
    envBackup = copy.deepcopy(env) # Make a copy of the environment to revert to
    for j in range(trainEpochs):
        print('epoch: ', j); sys.stdout.flush();
        # Reset the environment, keeping the learned information (in the agent) 
        env = copy.deepcopy(envBackup)
        
        experiment = createExp(env, agent)
 
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
    trainEpochs = 2000; numTrain = 1; numIterPerTrain = 5
    numBatchToKeep = 3;
    dimState = 1 # Currently we only record the distance to the thermal center
    agent = createAgentCont(numAngs, dimState, numBatchToKeep, numIterPerTrain)
    agent = addEpsGreedExplorer(agent)

    # Set optimistic initial values
    # optVal = 2
    # optLocs = range(20)
    # maxEpochs = 100
    # agent = setInitEst(optVal, optLocs, agent,maxEpochs)   
    
    # Print initial Q estimates
    print('Some initial SA values:')
    printSAValsNet(agent.module,range(20)) 
    print('Hit ENTER to continue.')
    
    input()
    
    # Learning
    print('\n\n Begin learning.\n\n')
    sys.stdout.flush();   
    agent = learningCycle(env,agent, trainEpochs, numTrain, numIterPerTrain)
    
    print('Sample final SA values:')
    printSAValsNet(agent.module,range(20))  
    sys.stdout.flush();
    
    # Testing
    testNet(agent.learner, agent.module,env, maxPlaneStartDist, stepSize,numAngs,thermRadius)

# Use a neural network to store state action values
def netMain():
    maxPlaneStartDist = 2  # Starting plane distance from thermal, or the maximum such distance if random placement of plane is allowed
    numAngs = 2            # The number of directions in which the plane is allowed to move
    thermRadius = 2;        # Standard deviation of thermal
    stepSize = 0.5          # How far the plane moves on each interaction
    netBasedMethod(maxPlaneStartDist,numAngs,thermRadius,stepSize)

if __name__ == "__main__":  
    netMain()


