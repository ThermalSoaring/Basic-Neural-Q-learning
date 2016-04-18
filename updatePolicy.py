import evalPolicy as ep

# Make policy greedy with respect to current value net 
def makeGreedy(valNet, polNet, policyEvalStates, numAct, stepSize, thermRadius):

    from pybrain.datasets import SupervisedDataSet  
    from pybrain.utilities import one_to_n
    
    supervised = SupervisedDataSet(polNet.indim, numAct) # numInput, numOutputs   
    
    # Try all the actions and see which has the best value    
    for state in policyEvalStates:
        actBest = 0
        vBest = -100000
        for action in range(numAct):            
            nextState = ep.updateState(state, stepSize, numAct, action, thermRadius)
            vNext = valNet.activate(nextState)
            if (vNext > vBest):
                actBest = action
                vBest = vNext        
        supervised.addSample(state, one_to_n(actBest, numAct))
    
    # Print supervised training set 
    # print(supervised)
    # input()
    
    # Train neural network
    # Keep interpolation here (good for smoothing)
    from pybrain.supervised.trainers.rprop import RPropMinusTrainer                
    trainer = RPropMinusTrainer(polNet, dataset=supervised, verbose=False)  
    numTrainIter = 30
    for i in range(numTrainIter):
        trainer.train()
    #trainer.trainUntilConvergence(maxEpochs=50) # Uses validation data, not as exact, but avoids overfitting
    return polNet