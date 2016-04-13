import numpy as np
from math import pi, cos, pow, sin, sqrt, exp

# Updates distance from center of thermal
def updateDist(oldDist, stepSize, numAct, chosenAction):
    # If 11 actions, 10 are direction movement actions -> action 10 is orbit
    if (chosenAction == numAct - 1):
        newDist = oldDist # Orbiting
    else:
        numAng = numAct - 1 # Subtract orbit command
        theta = chosenAction/(numAng-1)*pi;         

        deltaTempX = oldDist - stepSize*cos(theta)
        deltaTempY = sin(theta)*stepSize
        newDist = sqrt(pow(deltaTempX,2)+ pow(deltaTempY,2))
    return newDist

# Update height
def updateHeight(oldHeight, distToThermal, thermRadius):     
    # Gaussian shaped thermal
    #sigma = thermRadius
    #thermBoost = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal,2)/(2*pow(sigma,2))) 
    
    # Ring shaped thermal
    drop = 0.05 # Realize the plane will tend to fall
    shift = 5; # Where to put the center of boost, relative to center of thermal
    sigma = thermRadius
    thermBoost = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal-shift,2)/(2*pow(sigma,2)))   

    return (oldHeight + thermBoost - drop)
    
# Update both distance and height
def updateState(oldState, stepSize, numAct, chosenAction, thermRadius):
    oldDist = oldState[0]
    newDist = updateDist(oldDist, stepSize, numAct, chosenAction)
    
    oldHeight = oldState[1]
    newHeight = updateHeight(oldHeight, oldDist, thermRadius)
    
    return [newDist, newHeight]   
    
def getReward(state, scale):
    # Give a reward corresponding to our height    
    currHeight = state[1]
    return scale*currHeight
       
# Make values consistent with policy
def evalPolicy(valNet,polNet,policyEvalStates, vMaxAll, stepSize, thermRadius):
    vDiffStart = 10000
    vDiff = vDiffStart    
    while(vDiff > vMaxAll):
        vDiff = vDiffStart
        
        for state in policyEvalStates: # Go through the states in the discretization 
        
                # Stores next state according to the current policy
                nextState = [];

                # Determine what the chosen action is, from the policy network
                actionPref = polNet.activate(state)               
                chosenAction = np.argmax(actionPref) # Choose the one with highest output   
                
                # Determine the next state (from contThermalEnvironment)
                numAct = len(actionPref)                
                nextState = updateState(state, stepSize, numAct, chosenAction, thermRadius)
                                
                # Calculate reward given for transition                
                
                # Calculate new value of states under the current policy, based on reward given
                # Discount rate is how farsighted we are (between 0 and 1, with 1 being very far sighted, and 0 being not far sighted)
                discRate = 0.7
                scale = 1 # Size of reward
                reward = getReward(nextState, scale)    
                
                # Calculate new estimate for value 
                VstateNew = reward + discRate*valNet.activate(nextState);       
                
                # Determine how much the value changed
                # Keep track of maximum change seen so far
                VstateOld = valNet.activate(state)
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
                        tgt = valNet.activate(loc)
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
                # print('Old state:', state)
                # print('Preferences:', actionPref)
                # print('Choice:', chosenAction)
                # print('New state:', nextState)
                # print('Reward:', reward)
                # print('New Value:', VstateNew)
                # print('Value change:', vChange)
                # print('Max change:', vDiff)
                # print('Supervised data set:', supervised)
                
                # print('Actual network outputs:')
                # for loc in policyEvalStates:
                    # print(valNet.activate(loc))
                
                # input()
                
                # Return updated vallue function
        print('Max value change: ', vDiff)
        import sys ;sys.stdout.flush()
        
    return valNet    
    
    
    
