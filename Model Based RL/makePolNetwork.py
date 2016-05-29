from pybrain.structure import SigmoidLayer, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork

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

