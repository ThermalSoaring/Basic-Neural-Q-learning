import numpy as np
from math import pi, cos, pow, sin, sqrt, exp
from pybrain.datasets import SupervisedDataSet 
from pybrain.supervised.trainers.rprop import RPropMinusTrainer 

# Import to:
# -allow exiting with an error message
# -force printing
import sys 

''' Change the direction of the UAV
0 == facing center of thermal
1 == facing away from center of thermal

oldDir = the direction the UAV was facing prior to the function call
'''
def swapDir(oldDir):
    if (oldDir == 1):
        return 0
    elif (oldDir == 0):
        return 1
    else:        
        sys.exit("Invalid direction: should be 0 or 1.") 

''' Update distance and direction of UAV from center of thermal, based on the movement action chosen
This update occurs last (after height and direction are updated)

oldDist =       the distance of the UAV from the center of the thermal prior to the function call
oldDir =        the direction the UAV was facing prior to the function call
stepSize =      distance UAV moves upon making a non-orbiting action
action =        the action chosen, the effect of which on state variables is being carried out
                action = 0 -> UAV moves towards thermal center
                action = 1 -> UAV moves away from thermal center
                action = 2 -> UAV orbits (position from center of thermal remains unchanged)
'''   
def updateDist(oldDist, stepSize, action, oldDir):
    # If orbiting, distance from center doesn't change
    if (action == 2): 
        newDist = oldDist 
    else:  
        # UAV moves towards center of thermal
        if (action == 0):
            newDist = oldDist - stepSize
        # UAV moves away from center of thermal
        elif (action == 1): 
            newDist = oldDist + stepSize
        else: #  Invalid action    
            sys.exit("Invalid action provided: should be 0, 1, or 2.")   
    # If ther UAV passes through the center of the thermal, newDist will be negative
    if (newDist < 0):
        newDist = -newDist          # Work only with positive distances
        newDir = swapDir(oldDir)    # The direction of the UAV relative to the thermal center has changed
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
    
def getReward(state):
    # Give a reward corresponding to our height    
    currHeight = state[1]
    return currHeight
 
# Returns the action chosen by the policy net polNet in the state currState
def getPolAction(polNet, currState):
    actionPref = polNet.activate(currState) # List of preferences for each action, each in [0,1]               
    chosenAction = np.argmax(actionPref)    # The policy chooses the action with the highest value 
    return chosenAction
    
''' Make values consistent with policy
Create new value estimate at each of policyEvalStates.
-The new value estimate is
    :the reward obtained under the current policy 
    +
    the estimated value of the state landed in (discounted because in the future)
-Note that all value estimates are updated at the same time

See https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node41.html for details.

valNet =            value network (inputs = state variable values, output = value of state)
polNet =            policy network (inputs = state variable values, output = preferences for different actions)
policyEvalStates =  discretized states used for training networks
vMaxAll =           upper bound on maximum change in value estimates across all policyEvalStates before value function is considered self consistent
stepSize =          distance UAV moves upon making a non-orbiting action
thermRadius =       standard deviation of Gaussian shaped therma
discRate =          how farsighted we are (0 = future gain is worthless, 1 = future gain is just as important as present gain)  
numMaxEpochs =      IF USING VALIDATION DATA: maximum number of epochs to train the value network 
             =      IF NOT USING VALIDATION DATA: number of times to train the value network
'''
def evalPolicy(valNet,polNet,policyEvalStates, vMaxAll, discRate, numMaxEpochs, stepSize, thermRadius):
    
    # Ensure we update the value estimates at least once
    vDiffStart = 10000
    vDiff = vDiffStart   
    
    # Make sure vMaxAll is positive and less than the large vDiffStart
    if (vMaxAll <= 0 or vMaxAll >= vDiffStart):
        sys.exit("vMaxAll is too large or too small.") 
    
    while(vDiff > vMaxAll):
        vDiff = 10000   # Ensure the value estimates update at least once
        
        # Create set of data to hold new value estimates, for each of the discretized state estimates
        # Inputs = values of state variables (distance, height, direction)
        # Output = value of state described by value of state variables on inputs (a real number)
        supervised = SupervisedDataSet(valNet.indim, 1) # numInput, numOutputs   
        
        # For each state in the special list of discretized states, get a new value estimate
        for state in policyEvalStates:        
            # Stores next state according to the current policy
            nextState = []

            # Determine the action chosen by the policy net in this state
            # state -> (chosen action) 
            chosenAction = getPolAction(polNet, state)
            
            # Determine the next state, as well as its estimated value
            # state -> (chosen action) -> nextState
            nextState = updateState(state, stepSize, chosenAction, thermRadius)
            vNextState = valNet.activate(nextState)
                            
            # Calculate reward given for transition that occurs under current policy choice    
            # state -> (chosen action) -> nextState...results in reward           
            reward = getReward(nextState) 
            
            # Calculate new value of state under the current policy, based on reward given and value of state landed in                
            # discRate = How farsighted we are (0 = future gain is worthless, 1 = future gain is just as important as present gain)     
            vStateNew = reward + discRate*vNextState;       
            
            # To determine if convergence is occuring, determine how much value estimate changed
            # Keep track of maximum change seen so far (needs to be less than vMaxAll to allow us to move on)
            vStateOld = valNet.activate(state)
            vChange = abs(vStateOld - vStateNew)
            if (vDiff == vDiffStart): # Update vDiff to the maximum seen in this learning cycle (this while-loop iteration)
                vDiff = vChange
            elif (vChange > vDiff):
                vDiff = vChange
            
            # Store the new value estimate (used to train neural network once we have a new value estimate for all states)   
            # This means: "In this state, called 'state', we estimate a value of vStateNew."
            supervised.addSample(state, vStateNew) 
                        
        # Store these value estimates by updating the value network                           
        trainer = RPropMinusTrainer(valNet, dataset=supervised, verbose=False)               
                
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
                
        # Show how much the value estimates are changing, to illustrate convergence (or lack thereof)
        print('Max value change: ', vDiff)
        sys.stdout.flush()
        
    # Return updated value function  
    return valNet        