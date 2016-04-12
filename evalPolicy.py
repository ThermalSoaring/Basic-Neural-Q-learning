import numpy as np
from math import pi, cos, pow, sin, sqrt, exp

# Updates distance from center of thermal
def updateDist(oldDist, stepSize, numAng, chosenAction):
    theta = chosenAction/(numAng-1)*pi;         

    deltaTempX = oldDist - stepSize*cos(theta)
    deltaTempY = sin(theta)*stepSize
    newDist = sqrt(pow(deltaTempX,2)+ pow(deltaTempY,2))
    return [newDist]

# Gives reward of state transition
def getReward(state, thermRadius, scale):
    # Give a reward corresponding to how close to the thermal we are
    # The reward follows  a normal distribution with the standard deviation of the thermal
    distToThermal = state        
    sigma = thermRadius
    reward = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal,2)/(2*pow(sigma,2)))  
    return scale*reward

# Make policy greedy with respect to current value net
def makeGreedy(valNet, polNet, policyEvalStates, numAct, stepSize):

    from pybrain.datasets import SupervisedDataSet                
    supervised = SupervisedDataSet(polNet.indim, numAct) # numInput, numOutputs   
    
    # Try all the actions and see which has the best value    
    for state in policyEvalStates:
        vBest = -100000
        for action in range(numAct):            
            nextState = updateDist(state, stepSize, numAct, action)
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
        
def evalPolicy(valNet,polNet,policyEvalStates, vMaxAll, stepSize, thermRadius):
    vDiffStart = 10000
    vDiff = vDiffStart    
    while(vDiff > vMaxAll):
        vDiff = vDiffStart
        
        for state in policyEvalStates: # Go through the states in question                                
                # Stores next state according to the current policy
                nextState = [];

                # Determine what the chosen action is, from the policy network
                actionPref = polNet.activate([state])               
                chosenAction = np.argmax(actionPref) # Choose the one with highest output

                # Determine the next state (from contThermalEnvironment)
                numAng = len(actionPref)
                oldDist = state
                nextState = updateDist(oldDist, stepSize, numAng, chosenAction)     
                
                # Calculate reward given for transition                
                
                # Calculate new value of states under the current policy, based on reward given
                # Discount rate is how farsighted we are (between 0 and 1, with 1 being very far sighted, and 0 being not far sighted)
                discRate = 0.7
                scale = 10 # Size of reward
                reward = getReward(state, thermRadius,scale)
                
                # Calculate new estimate for value 
                VstateNew = reward + discRate*valNet.activate(nextState);       
                
                # Determine how much the value changed
                # Keep track of maximum change seen so far
                VstateOld = valNet.activate([state])
                vChange = abs(VstateOld - VstateNew)
                if (vDiff == vDiffStart):
                    vDiff = vChange
                elif (vChange > vDiff):
                    vDiff = vChange
                
                # Update value network with new estimate, keeping everything else the same   

                # First, get training examples
                from pybrain.datasets import SupervisedDataSet                
                supervised = SupervisedDataSet(valNet.indim, 1) # numInput, numOutputs   
                supervised.addSample(state, VstateNew)
                for loc in policyEvalStates: # Go through all discretized states 
                    if (loc != state):
                        inp = loc
                        tgt = valNet.activate([loc])
                        supervised.addSample(inp,tgt)
                        
                # Next, train on these training examples
                from pybrain.supervised.trainers.rprop import RPropMinusTrainer                
                trainer = RPropMinusTrainer(valNet, dataset=supervised, verbose=False)               
                
                # Train manually, to avoid using validation data
                # trainer.trainUntilConvergence(maxEpochs=maxEpochsVal, validationProportion = 0)   # Requires validation data 
                # I don't mind overfitting this - just so long as generalization is OK (so far, seems OK)
                numTrainIter = 30
                for i in range(numTrainIter):
                    trainer.train()                

                # Print training status
                # print('Old dist:', oldDist)
                # print('Preferences:', actionPref)
                # print('Choice:', chosenAction)
                # print('New dist:', nextState)
                # print('Reward:', reward)
                # print('New Value:', VstateNew)
                # print('Value change:', vChange)
                # print('Max change:', vDiff)
                # print('Supervised data set:', supervised)
                
                # print('Actual network outputs:')
                # for loc in policyEvalStates:
                    # print(valNet.activate([loc]))
                
                # input()
                
                # Return updated vallue function
        print('Max value change: ', vDiff)
        import sys ;sys.stdout.flush()
        
    return valNet