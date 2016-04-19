import evalPolicy as ep
import numpy as np
import random

# Make policy greedy with respect to current value net 
def makeGreedy(valNet, polNet, policyEvalStates, numAct, stepSize, thermRadius):

    from pybrain.datasets import SupervisedDataSet  
    from pybrain.utilities import one_to_n
    
    supervised = SupervisedDataSet(polNet.indim, numAct) # numInput, numOutputs   
    
    # Try all the actions and see which has the best value    
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
        bestVal = np.max(valList)
        # print('Val list: ', valList)
        
        # Choose the best action, breaking ties randomly
        bestActions = [i for i,j in enumerate(valList) if j == bestVal]     
        chosenAct = random.choice(bestActions)
        #print('Best actions:', bestActions)
        # print('Chosen action:', chosenAct)
        # import pdb; pdb.set_trace()
        supervised.addSample(state, one_to_n(chosenAct, numAct))
    
    # Print supervised training set 
    # print(supervised)
    # input()
    
    # Train neural network
    # Keep interpolation here (good for smoothing)
    print(supervised)
    
    
    from pybrain.supervised.trainers.rprop import RPropMinusTrainer                
    trainer = RPropMinusTrainer(polNet, dataset=supervised, verbose=False)  
    numTrainIter = 80
    for i in range(numTrainIter):
        trainer.train()
    #trainer.trainUntilConvergence(maxEpochs=50) # Uses validation data, not as exact, but avoids overfitting
    return polNet
