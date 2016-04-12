import evalPolicy as ep

# Make policy greedy with respect to current value net 
def makeGreedy(valNet, polNet, policyEvalStates, numAct, stepSize, thermRadius):

    from pybrain.datasets import SupervisedDataSet                
    supervised = SupervisedDataSet(polNet.indim, numAct) # numInput, numOutputs   
    
    # Try all the actions and see which has the best value    
    for state in policyEvalStates:
        vBest = -100000
        for action in range(numAct):            
            nextState = ep.updateState(state, stepSize, numAct, action, thermRadius)
            vNext = valNet.activate(nextState)
            if (vNext > vBest):
                actBest = action
                vBest = vNext
        from pybrain.utilities import one_to_n
        supervised.addSample(state, one_to_n(actBest, numAct))
    
    # Print supervised training set 
    # print(supervised)
    # input()
    
    # Train neural network
    from pybrain.supervised.trainers.rprop import RPropMinusTrainer                
    trainer = RPropMinusTrainer(polNet, dataset=supervised, verbose=False)  
    trainer.trainUntilConvergence(maxEpochs=50) # I'm OK with some interpolation here. It's the values we need to be exact on.
    return polNet