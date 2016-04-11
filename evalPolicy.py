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
    
def evalPolicy(valNet,polNet,policyEvalStates, vMaxAll, stepSize, thermRadius):
    vDiff = 10000
    while(vDiff > vMaxAll):
        for state in policyEvalStates: # Go through the states in question

                # Stores next state according to the current policy
                nextState = [];

                # Determine what the chosen action is, from the policy network
                actionPref = polNet.activate([state])               
                chosenAction = np.argmax(actionPref) # Choose the one with highest outpu

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
                maxEpochsVal = 70
                trainer = RPropMinusTrainer(valNet, dataset=supervised, batchlearning=True, verbose=False)
                trainer.trainUntilConvergence(maxEpochs=maxEpochsVal)    
                
                # print('Old dist:', oldDist)
                # print('Preferences:', actionPref)
                # print('Choice:', chosenAction)
                # print('New dist:', nextState)
                # print('Reward:', reward)
                # print('New Value:', VstateNew)
                # print('Supervised data set:', supervised)
                
                # print('Actual network outputs:')
                # for loc in policyEvalStates:
                    # print(valNet.activate([loc]))
                
                # input()

        # Determine the maximum change in V
        # vDiff = max(max(abs(V - vCopy)));
        # str = sprintf('Difference in V is %f', vDiff);
        # disp(str)
