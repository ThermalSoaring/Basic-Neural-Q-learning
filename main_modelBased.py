# Plan:
'''
 Develop a randomized policy (choose direction randomly)
 -- Effectively acheived by setting random values
 Calculate consistent state values for this policy.
 Develop a new policy that is greedy with respect to these state values.
 Loop.
 Each policy is guaranteed to be a strict improvement over the previous,
 except in the case in which the optimal policy has already been found.
'''
# We assume that we know the transition function
# -- Given state, action, we can determine the next state
# -- Later we want to relax this assumption (use probabilities)
# We assume that we know the reward function

import numpy as np

# Input: state variables
# Output: value of state
def createValNetwork(dimState, numHidden):
    # Build a feed forward neural network (with a single hidden layer)
    from pybrain.structure import SigmoidLayer, LinearLayer
    from pybrain.tools.shortcuts import buildNetwork
    valNet = buildNetwork(dimState, # Number of input units
                       numHidden, 	# Number of hidden units
                       1, 	        # Number of output units
                       bias = True,
                       hiddenclass = SigmoidLayer,
                       outclass = LinearLayer # Allows for a large output
                       )	
    return valNet

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
    
def mainModelBased():
    # 1. Create a random value function
    dimState = 1 # Number of state variables
    numHiddenVal = 20 # Number of hidden neurons
    valNet = createValNetwork(dimState,numHiddenVal)    
    
    # 2. Create a random policy network
    numHiddenPol = 20
    numAct = 4
    polNet = createPolNetwork(dimState, numHiddenPol, numAct)    
    
    # 3. Update values based on current policy
    # The policy is greedy with respect to the value estimates
    
    # 3a. Determine subset of state to update on
    # It isn't practical to visit every possible state
    # Instead, we will work on a discretized version of the state space, and trust to the neural network for interpolation    
    # Here we set the states used to update the policy
    start = 0
    stop = 10    
    policyEvalStates = np.linspace(start, stop, num=10) # Will need to be extended for more state variables
    
    import evalPolicy
    vMaxAll = 0.05 # We require values to stop changing by any more than this amount before we return the updated values
    stepSize = 0.5 # How far the airplane moves each time (needed to predict next state)
    thermRadius = 3 # Standard deviation of normal shaped thermal
    evalPolicy.evalPolicy(valNet,polNet,policyEvalStates,vMaxAll, stepSize, thermRadius)
    
    # 4. Update policy based on current values 

mainModelBased()











