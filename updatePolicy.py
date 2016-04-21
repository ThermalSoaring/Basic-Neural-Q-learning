import evalPolicy as ep
import numpy as np
import random

# Input: state variables
# Output: classification of actions (one hot encoded)
def createPolNetwork(dimState, numHidden, numAct):
     # Build a feed forward neural network (with a single hidden layer)
    from pybrain.structure import SigmoidLayer, SoftmaxLayer
    from pybrain.tools.shortcuts import buildNetwork
    polNet = buildNetwork(dimState, # Number of input units
                       numHidden, 	# Number of hidden units
                       numAct, 	        # Number of output units
                       bias = True,
                       hiddenclass = SigmoidLayer,
                       outclass=SoftmaxLayer # Outputs are in (0,1), and add to 1
                       )	
    return polNet


# Make policy greedy with respect to current value net 
def makeGreedy(valNet, polNet, policyEvalStates, numAct, stepSize, thermRadius,numHidden):

    from pybrain.datasets import SupervisedDataSet  
    from pybrain.utilities import one_to_n
    
    # Reset the policy network (keep same shape, but we don't want weights to grow large)
    polNet = createPolNetwork(polNet.indim, numHidden, numAct)
    
    supervised = SupervisedDataSet(polNet.indim, numAct) # numInput, numOutputs   
    
    # Try all the actions and see which has the best value     
    nextStateList = []
    nextValList = []
    actList = []
    for state in policyEvalStates:
        actBest = 0
        valList = []
        for action in range(numAct):            
            nextState = ep.updateState(state, stepSize, numAct, action, thermRadius)
            
            print('New state: ', nextState)
            # print('Action: ', action)
            # print('New state: ', nextState)            
            vNext = valNet.activate(nextState)
            valList.append(vNext)        
            
            # Store all states and values used, for debugging purposes            
            nextStateList.append(nextState)
            nextValList.append(vNext)
        bestVal = np.max(valList)
        # print('Val list: ', valList)
        
        # Choose the best action, breaking ties randomly
        bestActions = [i for i,j in enumerate(valList) if j == bestVal]     
        chosenAct = random.choice(bestActions)
        actList.append(chosenAct) # Keep track of all chosen actions for debugging purposes
        #print('Best actions:', bestActions)
        # print('Chosen action:', chosenAct)
        # import pdb; pdb.set_trace()
        supervised.addSample(state, one_to_n(chosenAct, numAct))
    
    # Print supervised training set 
    # print(supervised)
    # input()
    
    # Train neural network
    # Currently not using interpolation
    # Relying on resetting netowrk for smoothing
    # print(supervised)   
    from pybrain.supervised.trainers.rprop import RPropMinusTrainer                
    trainer = RPropMinusTrainer(polNet, dataset=supervised, verbose=True)  
    numTrainIter = 50
    for i in range(numTrainIter):
        trainer.train()
    #trainer.trainUntilConvergence(maxEpochs=50) # Uses validation data, not as exact, but avoids overfitting
    return (polNet, nextStateList, nextValList, actList) # Return both the updated net, and the data we trained on
