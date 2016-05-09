import numpy as np
from math import pi, cos, pow, sin, sqrt, exp
from pybrain.datasets import SupervisedDataSet 
from pybrain.supervised.trainers.rprop import RPropMinusTrainer 

# Change the direction of the UAV
# 0 == facing center of thermal
# 1 == facing away from center of thermal
def swapDir(oldDir):
    if (oldDir == 1):
        return 0
    elif (oldDir == 0):
        return 1
    else:
        import sys
        sys.exit("Invalid direction: should be 0 or 1.") 

# Updates distance from center of thermal (ideally done by simulator)
# Also updates direction, if we pass through the thermal center
def updateDist(oldDist, stepSize, chosenAction, oldDir):
    # Travel in a straight line
        # Direction given by oldDir
        # Distance given by stepSize

    # Actions 0 and 1 are going towards thermal and away. Action 2 is orbiting.
    if (chosenAction == 2): # If orbiting, distance from center doesn't change
        newDist = oldDist 
    else:  
        if (chosenAction == 0): # Go towards center
            newDist = oldDist - stepSize
        elif (chosenAction == 1): # Go away from center
            newDist = oldDist + stepSize
        else: # Throw an error
            import sys
            sys.exit("Invalid action provided: should be 0, 1, or 2.")   
    # It's possible to get a negative distance if we overshot the center
    # Note that we have change whether we are facing the thermal or no in this case
    if (newDist < 0):
        newDist = -newDist
        newDir = swapDir(oldDir)
    else:
        newDir = oldDir
            
    return (newDist, newDir)

# Update height
def updateHeight(oldHeight, distToThermal, thermRadius):     
    # Gaussian shaped thermal
    # sigma = thermRadius
    # thermBoost = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal,2)/(2*pow(sigma,2))) 
    
    # Ring shaped thermal
    drop = 0.05 # Realize the plane will tend to fall
    
    shift = 5; # Where to put the center of boost, relative to center of thermal
    sigma = thermRadius
    thermBoost = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal-shift,2)/(2*pow(sigma,2)))   

    return (oldHeight + thermBoost - drop)

# Update height and direction based on direction change
def updateDir(oldState, chosenAction): # chosenAction will be 0 (go towards thermal) or 1 (go away from thermal)
    newState = list(oldState)
    oldDir = oldState[2]
    switchPenal = 0.3 # Penalty for switching direction (in terms of height lost)
    # Penalize change in direction
    # State order is this: [pos, height, towardsCent]
    if (oldDir != chosenAction): # If a change of direction, reduce height
        newState[1] = oldState[1] - switchPenal
    # Update direction if not orbiting, otherwise keep direction the same
    if (chosenAction != 2):    
        newState[2] = chosenAction # Update direction
    return newState
    
# Update distance, height and direction
def updateState(oldState, stepSize, chosenAction, thermRadius):    
    # Update direction, and take the height penalty for changing direction
    # State order is this: [pos, height, towardsCent]
    tempState = updateDir(oldState, chosenAction)
    tempDist = tempState[0]
    tempHeight = tempState[1]  
    tempDir = tempState[2]
    
    # Update height  
    newHeight = updateHeight(tempHeight, tempDist, thermRadius)

    # Update distance
    (newDist, newDir) = updateDist(tempDist, stepSize, chosenAction, tempDir)  
    newState = [newDist, newHeight, newDir]
    
    return newState  
    
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
                nextState = updateState(state, stepSize, chosenAction, thermRadius)
                                
                # Calculate reward given for transition                
                
                # Calculate new value of states under the current policy, based on reward given
                # Discount rate is how farsighted we are (between 0 and 1, with 1 being very far sighted, and 0 being not far sighted)
                discRate = 0.7
                scale = 1 # Scaling of reward size
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
                supervised = SupervisedDataSet(valNet.indim, 1) # numInput, numOutputs   
                supervised.addSample(state, VstateNew)
                for loc in policyEvalStates: # Go through all discretized states 
                    if (loc != state):
                        inp = loc
                        tgt = valNet.activate(loc)
                        supervised.addSample(inp,tgt)
                        
                # Next, train on these training examples                               
                trainer = RPropMinusTrainer(valNet, dataset=supervised, verbose=False)               
                
                # Train manually, to avoid using validation data
                # trainer.trainUntilConvergence(maxEpochs=50)   # Requires validation data 
                # I don't mind overfitting this - just so long as generalization is OK (so far, seems OK)
                # Sometimes we do get weird curves where there should not be
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
        # Show how much the value network is changing
        print('Max value change: ', vDiff)
        import sys ;sys.stdout.flush()
        
    return valNet    
    
    
    
