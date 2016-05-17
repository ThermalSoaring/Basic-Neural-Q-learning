# Import:
import numpy as np

# To force printing
import sys;

# For timestamp on debugging images 
import time     
import datetime

# To allow folder creation, for saving debug images
import os

# To take product of state variables
import itertools

# For making the value estimates self consistent
import evalPolicy

# For making the policy greedy with respect to the current value estimates
import updatePolicy

# For creating the value and policy networks
from pybrain.structure import SigmoidLayer, LinearLayer, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork

# Import libraries for creating and saving plots
import matplotlib as mpl
mpl.use('Agg') # Allows plots to be saved instead of being displayed
import matplotlib.pyplot as plt             

''' Returns a value network
dimState =  number of state variables
numHidden = number of neurons in hidden layer
 ''' 
def createValNetwork(dimState):
    # Build a feed forward neural network (with one hidden layer)
    # Input of policy network: value of state variables
    # Output of policy network: value of the state described
    if (dimState >= 2):
        valNet = buildNetwork(dimState, # Number of input units
                           20,          # Number of hidden units 
                           1, 	        # Number of output units
                           bias = True,
                           hiddenclass = SigmoidLayer,
                           outclass = LinearLayer # Allows for a large output (not just between 0 and 1)
                           )	
        return valNet
    else:
        sys.exit('Invalid number of state variables. Should be >= 2.')

''' Returns a policy network
dimState =  number of state variables
numHidden = number of neurons in hidden layer
numAct =    number of possible actions
'''
def createPolNetwork(dimState, numAct):
    # Build a feed forward neural network (with one hidden layer)
    # Input of policy network: state variables
    # Output of policy network: preference towards each action
    if (dimState >= 1 and numAct >= 1):
        polNet = buildNetwork(dimState, # Number of input units
                           20, 	        # Number of hidden units
                           numAct, 	    # Number of output units
                           bias = True,
                           hiddenclass = SigmoidLayer,
                           outclass=SoftmaxLayer # Outputs are in the interval (0,1), and they sum to 1
                           )	
        return polNet
    else:
        sys.exit('Invalid number of state variables or actions. Both should be >= 1.')

''' Plot estimated value function at a variety of heights and distances
valNet =            value network (inputs = state variable values, output = value of state)
evalDir =           list of directions we can travel in
policyEvalStates =  discretized states used for training networks
heightEval =        the heights at which the value function is evaluated
'''    
def graphValues(valNet,heightEval):    
    # Create list of values of distances from the thermal - x-axis values
    dist = np.linspace(0, 10, num=60)  
    
    # Lists of value estimates, to be plotted - y-axis values
    # There is one list per height
    valList = []    
    
    # Evalute value function at desired states
    heightInd = 0   # Index on heights - the height currently being examined      
    dirEval = 0  # Directions for which to evaluate value function (CURRENTLY LIMITED TO ONE DIRECTION AT A TIME)  
    for height in heightEval: 
        valList.append([])
        # Get the value given the state variable values (position, height and direction)
        for pos in dist:  
            if (valNet.indim == 2):
                val = valNet.activate([pos, height])[0] 
            elif (valNet.indim == 3):
                val = valNet.activate([pos, height, dirEval])[0] 
            else:
                sys.exit('Invalid input layer size on value net. Should be 2 or 3.')
            
            # Append the value to the value list for the current height
            valList[heightInd].append(val)   
            
        # Plot values at current height
        plt.plot(dist,valList[heightInd], label = height)
        
        # Move to next height for value function evaluation
        heightInd = heightInd + 1
    
    # Set axes labels and x-axis limits
    plt.xlabel('Distance')
    plt.ylabel('Value')
    plt.xlim([0,10])
    plt.legend()
    # Indicate the direction the UAV is facing under which the policy is being evaluated
    plt.title('Value Function, Direction = ' + str(dirEval)) 
    
  
''' Plot policy over a range of distances, for a fixed height and direction
-Plots preference for each action
-Plots most recent policy training examples

polNet =            policy network (inputs = state variable values, output = preferences for different actions)
policyEvalStates =  discretized states used for training networks
actList =           the best actions under the current value function - used as training examples to form this policy
'''
def graphPolicy(polNet, policyEvalStates, actList):

    # Create list of values of distances from the thermal - x-axis values
    dist = np.linspace(0, 10, num=60) 
    
    # Lists of preference values, one list per action - y-axis values
    prefToward = []
    prefAway = []
    preOrb = []
    
    heightEval = 0  # Heights at which to evalute the policy network
    dirEval = 0     # Direction at which to evaluate the policy network
    for pos in dist:
        # Get policy preference towards each action
        if (polNet.indim == 2):
            preferences = polNet.activate([pos, heightEval])
        elif (polNet.indim == 3):
            preferences = polNet.activate([pos, heightEval, dirEval])
        else:
            sys.exit('Invalid input layer size on policy net. Should be 2 or 3.')   
        
        # Save the policy preferences in the appropriate lists
        if (polNet.outdim >= 2):
            prefToward.append(preferences[0])   # Move towards thermal
            prefAway.append(preferences[1])     # Move away from thermal    
        if (polNet.outdim >= 3):
            preOrb.append(preferences[2])       # Orbit thermal
        if (polNet.outdim > 3):
            sys.exit('Invalid output layer size on policy net. Should be 1, 2, or 3.')
        
    # Plot preferences of current policy over distances dist
    plt.plot(dist,prefToward, label = 'Move towards')
    plt.plot(dist,prefAway, label = 'Move away')
    plt.plot(dist,preOrb, label = 'Orbit')
    #plt.legend()
        
    # Set axes labels and limits
    plt.xlabel('Distance')
    plt.ylabel('Preference')  
    plt.xlim([0,10])
    plt.ylim([-0.1,1.1])
    
    # Indicate height and direction used
    if (polNet.indim == 2):
        plt.title('Policy at h=' + str(heightEval))
    elif (polNet.indim == 3):
        plt.title('Policy at h=' + str(heightEval) + ', d=' + str(dirEval))
        
    # Plot where the training examples are using vertical lines
    for state in policyEvalStates:
        posOfState = state[0]
        plt.axvline(posOfState)
    
    # =============================================================================
    
    # Plot the actions used as training examples to form this policy  
    # Only plot at height and distance used for plotting the policy above    
    trainDist = []      # Distance from center of thermal training example was at - x-axis value
    trainChoice = []    # Choice in training example - y-axis value  
    
    stateInd = 0     # Keeps track which state we are currently examining
    for state in policyEvalStates:      
        # Check that state is at desired height and direction
        if ((polNet.indim == 2 and state[1] == heightEval) or (polNet.indim == 3 and state[1] == heightEval and state[2] == dirEval)): 
            # Get distance from center of thermal in this state - x-axis data
            trainDist.append(state[0])
        
            # Get best action under value network in this state and append to plotted y-axis data
            action = actList[stateInd] 
            trainChoice.append(action)
        
        # Move to the next state
        stateInd = stateInd + 1
    
    # Plot the actions used as training examples to form this policy
    plt.plot(trainDist, trainChoice, 'o')
 
''' Returns a time stamp in the form m-d H-M-S
'''
def createTimeStamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d %H-%M-%S')
    return st

''' Prints current policy to console
polNet =            policy network (inputs = state variable values, output = preferences for different actions)
policyEvalStates =  discretized states used for training networks
'''
def printPolicy(polNet, policyEvalStates):
    print('Policy:')
    print('Choices \t Discretizing States')
    for state in policyEvalStates:
        # The action chosen is the one with the highest preference under the current policy
        action = np.argmax(polNet.activate(state))
        
        # Print the value of the state variables
        print(action, end = '\t\t')        
        for stateVar in state:
            print('%.2f' % stateVar, end = "\t")            
        print()
        
        # Force printing
        sys.stdout.flush()

''' Prints current value estimates to console
valNet =            value network (inputs = state variable values, output = value of state)
policyEvalStates =  discretized states used for training networks
'''
def printVal(valNet, policyEvalStates):
    #Print update value network on a few sample points
    print('Updated value function:')
    print('Value \t State')
    for state in policyEvalStates:
        print(valNet.activate(state), '\t', state) 
 
''' Main program loop
-Create a randomized policy (randomly maps values of state variables to preferences)
-Calculate consistent state values under this policy
-Develop a new policy that is greedy with respect to the new state values
-Repeat. Every severl repetitions, plot the current value and policy network
'''  
def mainModelBased():
    # Time stamp this run:
    st = createTimeStamp()
    
    # Create a value function (randomly initialized values)    
    # ===========================================================================
    # Number of state variables used in discretized training examples 
    dimState = 3 # 1 = Position from thermal, 2 = height, 3 = direction
    valNet = createValNetwork(dimState)
    # ===========================================================================    
    
    # Create a policy network (randomly initialized policy choices)
    # ===========================================================================
    numAct = 3 # Number of actions (move towards thermal, away from thermal, orbit)
    polNet = createPolNetwork(dimState, numAct)   
    # ===========================================================================    
    
    # Choose discretizing states
    # These are the states at which:
        # The value function makes sures it is self consistent
        # The policy network makes sure it is greedy
    # ===========================================================================    
    if (dimState >= 2):    
        # Distance discretization
        evalDist = np.linspace(0, 10, num=10)
        # Height discretization
        evalHeight = [0, 0.5]#[0, 0.5] # 
    if (dimState >= 3):
        # Direction discretization 
        # 0 = towards thermal center, 1 = away from thermal center
        evalDir = [0]  
    
    # Takes cartesian product to create complete discretized states
    if (dimState == 2):
        policyEvalStates = list(itertools.product(evalDist,evalHeight))
    elif (dimState == 3):
        policyEvalStates = list(itertools.product(evalDist,evalHeight, evalDir)) 
    else:
        sys.exit('Wrong number of state dimensions; should be 1,2 or 3.')        
    # =========================================================================== 
    
    # Print initial (random) policy
    printPolicy(polNet, policyEvalStates)    
    
    # Specify environmental parameters
    # =========================================================================== 
    stepSize = 0.1  # Distance UAV moves upon making a non-orbiting action
    thermRadius = 3 # Standard deviation of Gaussian shaped thermal 
    thermCenter = 0 # Center of Gaussian shaped thermal
    switchPenal = 0.2 # the height lost for chosing to switch directions (without passing through center of thermal)
    # =========================================================================== 

    # Specify learning parameters
    # ===========================================================================
    # For making value function consistent:
    vMaxAll = 0.5   # Upper bound on maximum change in value estimates across all policyEvalStates before value function is considered self consistent
    discRate = 0.7  # How farsighted we are (0 = future gain is worthless, 1 = future gain is just as important as present gain)  
    numMaxEpochsVal = 30 # IF USING VALIDATION DATA: maximum number of epochs to train the value network 
                         # IF NOT USING VALIDATION DATA: number of times to train the value network
    
    # For the overall learning process"
    numLearn = 100  # Number of times to repeat the following process: make value estimates consistent, make policy greedy with respect to new values
    
    # ===========================================================================             
    
    # Learning loop
    # =========================================================================== 
    for i in range(numLearn):  
        # Display current learning iteration
        print('Start iteration: ', str(i))
        
        # Make value estimates consistent with current policy        
        valNet = evalPolicy.evalPolicy(valNet,polNet,policyEvalStates,vMaxAll,discRate, numMaxEpochsVal, stepSize, thermRadius, thermCenter,switchPenal)  
        
        # Make policy greedy with respect to current value estimates
        # TEMP
        numHiddenPol = 20 # Required because policy network is being recreated everytime - this should not be necessary!
        # END TEMP
        (polNet, nextStateList, nextValList, actList) = updatePolicy.makeGreedy(valNet, polNet, policyEvalStates,numAct,stepSize, thermRadius, thermCenter, numHiddenPol, switchPenal)
        
        # Print updated value function
        printVal(valNet, policyEvalStates)
            
        # Print updated policy
        printPolicy(polNet, policyEvalStates) 
                
        # Plot the current value function and policy
        # =========================================================================== 
        if (i == 0):       
            plt.figure()      
        
        # Only create and save a plot every few iteration
        savePeriod = 5 # How many iterations to wait between creating a plot
        if (i % savePeriod == 0):
            # Clear the plot
            plt.clf() 
                       
            # Plot value network 
            plt.subplot(2, 1, 1)
            heightEval = [0.5, 0.2, 0, -0.2, -0.5] # Heights at which to evalute value function
            graphValues(valNet,heightEval)
            
            # Plot policy network
            plt.subplot(2, 1, 2)
            graphPolicy(polNet, policyEvalStates, actList)   
        
            # Store photos
            # Include a time stamp and the current iteration 
            saveDir = '/home/david.egolf/debugImages/' + st + '/' # Write to a time stamped folder
            if not os.path.isdir(saveDir): # Make the directory if needed
                os.makedirs(saveDir)                
            plt.savefig(saveDir + 'iter = ' + str(i))  # Indicate the curren iteration
        # =========================================================================== 
        
    # =========================================================================== 

# Run the program
mainModelBased()