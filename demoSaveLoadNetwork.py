
# To save and load a network
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

# To build the network
from pybrain.tools.shortcuts import buildNetwork

# For printing the network
import updatePolicy as up

# Create a demo network
net = buildNetwork(2,4,1)

# Print demo network
print('Before saving:')
up.printNet(net)

# Save the demo network
fileName = 'demoNet.xml'
NetworkWriter.writeToFile(net, fileName)

# Read demo network, and print
net = NetworkReader.readFrom(fileName) 
print('\n\nAfter loading:')
up.printNet(net)