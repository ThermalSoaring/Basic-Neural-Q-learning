import numpy as np

# For making the value estimates self consistent
import evalPolicy as ep

# For breaking ties randomly between actions with the same value
import random

# For the creation of a policy network
import makePolNetwork as mp

# Allow for "resilient backpropogation" training on neural networks
# See here for details: https://en.wikipedia.org/wiki/Rprop
from pybrain.datasets import SupervisedDataSet  
from pybrain.supervised.trainers.rprop import RPropMinusTrainer

# For creating one-hot encoded training examples
# For examples, if there are three actions, and we want the second one the desired output of the policy network is:
# policy network(state) = (0 1 0) (no preference for 1st and 3rd, chooses 2nd)
from pybrain.utilities import one_to_n

''' Prints a neural network
Source of code: http://stackoverflow.com/questions/8150772/pybrain-how-to-print-a-network-nodes-and-weights
Goes through each layer and prints the weights
'''
def printNet(net):
    for mod in net.modules:
        print("Module:", mod.name)
        if mod.paramdim > 0:
            print("--parameters:", mod.params)
        for conn in net.connections[mod]:
            print("-connection to", conn.outmod.name)
            if conn.paramdim > 0:
                 print("- parameters", conn.params)
        if hasattr(net, "recurrentConns"):
            print("Recurrent connections")
            for conn in net.recurrentConns:
                print("-", conn.inmod.name, " to", conn.outmod.name)
                if conn.paramdim > 0:
                    print("- parameters", conn.params)


''' Makes policy network greedy with respect to current value estimate
At each of policyEvalStates, all the actions are tried
The action that lands in a state with highest value (by current value function) is chosen as desirable
These desirable actions are used as training examples to form the updated policy network

valNet =            value network (inputs = state variable values, output = value of state)
polNet =            policy network (inputs = state variable values, output = preferences for different actions)
policyEvalStates =  discretized states used for training networks
numAct =            number of actions (move towards thermal, away from thermal, orbit)
stepSize =          distance UAV moves upon making a non-orbiting action
thermRadius =       standard deviation of Gaussian shaped thermal
thermCenter =       center of Gaussian shaped thermal
switchPenal =       the height lost for choosing to switch directions (without passing through center of thermal)
numMaxEpochs =      IF USING VALIDATION DATA: maximum number of epochs to train the policy network 
                    IF NOT USING VALIDATION DATA: number of times to train the policy network    
'''
# 
def makeGreedy(valNet, polNet, policyEvalStates, numAct, stepSize, thermRadius,thermCenter, switchPenal, numMaxEpochs):    
    
    # Create a new (randomly initialized) policy net, so as to have small weights
    # Training the old net repeatedly results in very large weights - this should be avoidable, though
    polNet = mp.createPolNetwork(polNet.indim, numAct)
    
    # Creates a list to hold training examples
    # The inputs are the values of the state variables
    # The outputs are the preference towards each of the numAct actions
    supervised = SupervisedDataSet(polNet.indim, numAct) # numInputs = polNet.indim, numOutputs = numAct
    
    # For each state in policyEvalStates, determine the best action  
    actList = [] # The list of actions chosen, for debug purposes
    for state in policyEvalStates:
    
        # Clear the value list, because we are considering a new state
        valList = []
        
        # Consider all actions - which will be best?
        for action in range(numAct): 
            # Determine where we land if we try the action called "action"
            nextState = ep.updateState(state, stepSize, action, thermRadius, thermCenter, switchPenal)           
            
            # Determine the value of the state landed in, nextState
            vNext = valNet.activate(nextState)
            
            # Add the value to the list for this state
            valList.append(vNext)        
        
        # Choose the action that lands in the state with highest value
        # Breaks ties randomly
        bestVal = np.max(valList)       
        bestActions = [i for i,j in enumerate(valList) if j == bestVal]     
        chosenAct = random.choice(bestActions)
        
        # Store the action chosen, for debug display purposes
        actList.append(chosenAct) 
        
        # Store the new training example, using one-hot encoding
        # In state "state", we want a preference of 1 for chosenAct, and 0 for all others
        supervised.addSample(state, one_to_n(chosenAct, numAct))
    
    # Train the network, using resilient backpropogation
    trainer = RPropMinusTrainer(polNet, dataset=supervised, verbose=False)  
    
    # Two ways to update the neural network: (CHOOSE ONLY ONE)  
    # =========================================================================== 
    # 1. With validation data
    # --The network is trained on a subset of training examples, and then checked to see how well it generalizes to the unused examples
    # --Danger of underfitting (not matching all the training examples)
    trainer.trainUntilConvergence(maxEpochs=numMaxEpochs)
    
    # # 2. Without validation data
    # # --The network is trained on all the training examples.
    # # --Danger of overfitting (sacrificing interpolation)
    # for i in range(numMaxEpochs):
        # trainer.train()     
     # =========================================================================== 
    
    return (polNet,  actList) # Return both the updated net, and the data we trained on
