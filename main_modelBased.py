# LINUX VERSION

# Import:
import numpy as np

# For timestamp of file saved
import time     
import datetime

# To allow folder creation
import os

# To take product of state variables
import itertools

# For making the policy self consistent
import evalPolicy

# Input: values of state variables
# Output: value of state
def createValNetwork(dimState, numHidden):
    # Build a feed forward neural network (with one hidden layer)
    from pybrain.structure import SigmoidLayer, LinearLayer
    from pybrain.tools.shortcuts import buildNetwork
    valNet = buildNetwork(dimState, # Number of input units
                       numHidden,   # Number of hidden units 
                       1, 	        # Number of output units
                       bias = True,
                       hiddenclass = SigmoidLayer,
                       outclass = LinearLayer # Allows for a large output (not just between 0 and 1)
                       )	
    return valNet

''' Returns a policy network
Input: state variables
Output: classification of actions (one hot encoded)
'''
def createPolNetwork(dimState, numHidden, numAct):
    # Build a feed forward neural network (with one hidden layer)
    from pybrain.structure import SigmoidLayer, SoftmaxLayer
    from pybrain.tools.shortcuts import buildNetwork
    polNet = buildNetwork(dimState, # Number of input units
                       numHidden, 	# Number of hidden units
                       numAct, 	    # Number of output units
                       bias = True,
                       hiddenclass = SigmoidLayer,
                       outclass=SoftmaxLayer # Outputs are in the interval (0,1), and they sum to 1
                       )	
    return polNet

''' Plot estimated value function 
valNet = value network (inputs = state values, output = value of state)
evalDir = list of directions we can travel in
policyEvalStates = discretized states
'''
def graphValues(valNet, evalDir, policyEvalStates, nextStateList, nextValList, maxX):
    import matplotlib.pyplot as plt
    
    # Create distance linspace for plotting
    start = 0; stop = 10;
    dist = np.linspace(start, stop, num=60)  
    
    # Determine value at multiple heights
    valList = [] # List of value estimates
    heightInd = 0 # Which height we're working with
    for height in [0.5, 0.2, 0, -0.2, -0.5]: # Desired heights     
        valList.append([])
        for pos in dist:              
            towardsCent = 0 # Desired direction to evaluate value network at
            val = valNet.activate([pos, height, towardsCent])[0] 
            valList[heightInd].append(val)            
        plt.plot(dist,valList[heightInd], label = height)
        heightInd = heightInd + 1
    #plt.legend()
    plt.xlim([0,maxX])
    plt.xlabel('Distance')
    plt.ylabel('Value')
    plt.title('Value Function, Direction = ' + str(towardsCent))
    
    # Plot data used to train policy network
    trainValToPlot = []
    trainDistToPlot = []    
           
    # Plot data used for training policy network
    # State order is this: [pos, height, towardsCent]
    i = 0   # Which training example
    for state in nextStateList:
        towardsCent = 0 # Plot only training examples going towards thermal  
        heightVal = 0   # Plot only examples at zero height
        if (state[2] == towardsCent and state[1] == heightVal):
            trainValToPlot.append(nextValList[i])
            trainDistToPlot.append(state[0])
        i = i + 1
    plt.plot(trainDistToPlot, trainValToPlot, 'o')
    
    # Indicate where our training examples are
    for state in policyEvalStates:
        stateDist = state[0]
        plt.axvline(stateDist)
        
    # Graph saved to file elsewhere periodically
    
  
''' Plots current policy 
    -Plots preference for each action
    -Plots training examples for policy
'''
def graphPolicy(polNet, policyEvalStates, actList, maxX):
    import matplotlib.pyplot as plt
    start = 0; stop = 10;
    dist = np.linspace(start, stop, num=60) # Where to evaluate policy
    
    prefToward = []
    prefAway = []
    preOrb = []
    # Print how much we like the different actions
    heightVal = 0       # What height to evaluate the policy at
    towardsCentVal = 0  # What facing direction to evaluate the policy at
    for pos in dist:
        height = heightVal   
        towardsCent = towardsCentVal
        preferences = polNet.activate([pos, height, towardsCent])
        # Record preference for different options
        prefToward.append(preferences[0])  
        prefAway.append(preferences[1])      
        preOrb.append(preferences[2])   
    plt.plot(dist,prefToward, label = 'Move towards')
    plt.plot(dist,prefAway, label = 'Move away')
    plt.plot(dist,preOrb, label = 'Orbit')
    plt.xlabel('Distance')
    plt.ylabel('Preference')
    plt.xlim([0,maxX])
    plt.ylim([-0.1,1.1])
    
    plt.title('Policy at h=' + str(heightVal))
    plt.legend()

    # Indicate where our training examples are
    for state in policyEvalStates:
        stateDist = state[0]
        plt.axvline(stateDist)
    
    # Indicate the training example choices (stored in actList)
    trainChoice = []
    trainDist = []
    i = 0
    relActs = []    
    for state in policyEvalStates:  
        # Store only training examples for desired height and direction
        if (state[1] == heightVal and state[2] == towardsCentVal):
            trainChoice.append(actList[i])
            relActs.append(actList[i])
            trainDist.append(state[0])
        i = i + 1
    plt.plot(trainDist, trainChoice, 'o')
    # print('All acts: \n', actList)
    # print('Rel actlist: \n', relActs)    
    
''' Creates a list of states
    This are the states at which:
    -The value function makes sures it is self consistent
    -The policy makes sure it is greedy
    Use neural network interpolation to make estimates elsewhere
''' 
def createEvalStates(): 
    # Distance discretization
    start = 0; stop = 10;    
    evalDist = np.linspace(start, stop, num=8)
    
    # Height discretization
    evalHeight = [0, 0.5] # 
    
    # Direction discretization 
    # 0 = towards thermal center, 1 = away from thermal center
    evalDir = [0]  
    
    # Takes cartesian product to create complete discretized states
    policyEvalStates = list(itertools.product(evalDist,evalHeight, evalDir)) 
    
    return policyEvalStates

''' Returns a time stamp in the form m-d H-M-S
'''
def createTimeStamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d %H-%M-%S')
    return st

''' Returns a randomly initialized value function
'''
def initValNetwork():
    # Number of state variables used in discretized training examples (position from thermal, height, direction (towards or away center))
    dimState = 3 
    
    numHiddenVal = 20 # Number of neurons in hidden layer of value network
    valNet = createValNetwork(dimState, numHiddenVal)
    return valNet

''' Returns a randomly initialized policy network    
'''
def initPolNetwork(valNet):
    numHiddenPol = 20 # Number of neurons in hidden layer of policy network
    
    # Number of angles UAV can travel in  
    # numAng = 2 indicates UAV can travel away or towards thermal center
    numAng = 2        

    # Number of state variables
    dimState = valNet.indim
    
    numAct = numAng + 1 # Number of actions (additional action is for orbit - preserve distance from center)
    polNet = createPolNetwork(dimState, numHiddenPol, numAct) 
    
    return polNet
 
''' Prints current policy to console
'''
def printPolicy(polNet, policyEvalStates):
    print('Policy:')
    print('Choices \t Discretizing States')
    for state in policyEvalStates:
        print(np.argmax(polNet.activate(state)), end = '\t\t')        
        for stateVar in state:
            print('%.2f' % stateVar, end = "\t")            
        print()

    import sys; sys.stdout.flush()
  
''' Main program loop
Strategy:
-Develop a randomized policy (choose actions randomly)
-Calculate consistent state values for this policy.
-Develop a new policy that is greedy with respect to these state values.
-Loop. 
'''  
def mainModelBased():
    # Time stamp this run:
    st = createTimeStamp()

    # Create a value function (randomly initialized values)
    valNet = initValNetwork()
    
    # Create a policy network (randomly initialized policy choices)
    polNet = initPolNetwork(valNet)       
    
    # Choose discretizing states
    # Where value network checks for consistency, and policy network is greedy
    policyEvalStates = createEvalStates()   
    
    # Print initial policy
    printPolicy(polNet, policyEvalStates)
    
    # Go back and forth between value and policy
    vMaxAll = 0.5   # We require values to stop changing by any more than this amount before we return the updated values
    stepSize = 0.1  # How far the airplane moves each time (needed to predict next state, make policy decisions)
    thermRadius = 3 # Standard deviation of normal shaped thermal   
    
    numLearn = 100 # Number of times to repeat learning cycle
    
    # Learning iterations loop
    for i in range(numLearn):  
        print('Start iteration: ', str(i))
        # Make values consisstent with policy
        valNet = evalPolicy.evalPolicy(valNet,polNet,policyEvalStates,vMaxAll, stepSize, thermRadius)   
        
        # 4. Update policy based on current values 
        import updatePolicy
        
        # TEMP
        numAct = 3  # Towards, away, orbit
        numHiddenPol = 20 
        evalDir = [0] 
        maxX = 10 # Upper limit for plotting
        # END TEMP
        
        (polNet, nextStateList, nextValList, actList) = updatePolicy.makeGreedy(valNet, polNet, policyEvalStates,numAct,stepSize, thermRadius, numHiddenPol)
        
        # TEMP
        # if (printPolVal == 1):
            # Print update value network on a few sample points
            # print('Updated value function:')
            # print('Value \t State')
            # for state in policyEvalStates:
                # print(valNet.activate(state), '\t', state) 
        # END TEMP
            
        # Print updated policy
        printPolicy(polNet, policyEvalStates) 
                
        # Display the estimated value function and policy
        clearPeriod = 1
        if (i == 0):
            import matplotlib as mpl
            mpl.use('Agg') # Allows us to save image instead of displaying
            
            import matplotlib.pyplot as plt              
            plt.figure()              

        # Periodically clear plot
        if (i % clearPeriod == 0 and i != 0):
            plt.clf() 
                   
        # Plot value network
        # valList is the values used by the policy network
        plt.subplot(2, 1, 1)
        graphValues(valNet, evalDir,policyEvalStates, nextStateList, nextValList, maxX)
        
        # Plot policy network
        plt.subplot(2, 1, 2)
        graphPolicy(polNet,policyEvalStates, actList, maxX)        
        
        # Save an image of the results
        if (i % 5 == 0):
            # Store photos one directory up, with a time stamp and an i value
            saveDir = '/home/david.egolf/debugImages/' + st + '/'
            if not os.path.isdir(saveDir): # Make the directory if needed
                os.makedirs(saveDir)
            plt.savefig(saveDir + 'iter = ' + str(i))        

mainModelBased()
