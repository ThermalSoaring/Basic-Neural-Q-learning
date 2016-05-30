'''
Based on CartPoleEnvironment by Thomas Rueckstiess, ruecksti@in.tum.de
Provides sensor values and updates sensor values based on actions (uses Gaussian shaped thermal)
Returned sensor values are discretized (tells UAV what chunk of the world it is in)
 -Currently, only sensor value is distance from center of thermal
 
This discrete version used by the table based method
'''
from math import sin, cos, sqrt, pow, pi, floor

from pybrain.rl.environments.environment import Environment

class simpThermEnvironment(Environment):
    
    # The number of dimensions in the action vector
    # The input is cos(theta), where theta is the angle from zero along a line drawn from the plane to the center of the thermal
    # - To illustrate, the input is 1 if the plane travels directly towards the thermal, and 0 if it travels at right angles    
    indim = 1 
    
    # The number of sensor values the environment produces
    # Currently only sensor is distance from center of thermal
    outdim = 1

    def __init__(self, planeStartDist, stepSize, numAngles,numDist, thermRadius):
        # planeStartDist is the distance the plane starts from the thermal center
        self.planeStartDist = planeStartDist
        
        # stepSize is how far the UAV moves each time
        self.stepSize = stepSize
                
        # numAngles is the number of discrete directions the plane can move
        self.numAngles = numAngles
        
        # numDist is the number of chunks we discretize distance from center of thermal, for making Q estimates
        # A continuous distance value is used here in the environment class
        self.numDist = numDist
        
        # Sets standard deviation of normal shaped reward function
        self.thermRadius = thermRadius
        
        # Initialize the stored values for the states and actions of the agent
        self.reset() 
        self.action = 0.0  
        self.delay = False
        
    # Returns distance to center of thermal after action is carried out
    def getSensors(self):
        outBound = self.planeStartDist*1.5 # Beyond this distance, all values estimates get chunked together
        distToCent = self.sensors
        if (distToCent > outBound):
            distIndex = self.numDist - 1
        else:
            # Returns the chunk (discretized) that the plane is currently in, for purposes of making a Q estimates table
            distIndex = floor((distToCent/(outBound/(self.numDist-1))))
                  
        return [distIndex]

    # Carries out the action provided by the agent
    # The action is theta, where theta is the angle (in radians) from a line drawn from the plane to the center of the thermal
    # - To illustrate, the input is 0 if the plane travels directly towards the thermal, and pi/2 if it travels at a right angle to the thermal center    
    def performAction(self, action):   
        self.action = action 
        self.step() # Carry out an interaction

    # Update sensor values to capture the movement of the UAV, given the current intended action
    def step(self):
        # Get the current distance from the center
        oldDist = self.sensors

        # We need to convert the action into radians
        # Assume we have numAngles = n+1
        # Index 0           ->      0 radians
        # Index last (n)    ->      pi radians
        # Index a           ->      a/n*pi radians = a/(numAngles-1)*pi       
        theta = self.action/(self.numAngles-1)*pi;
        
        # Calculate new distance from center of thermal
        stepSize = self.stepSize                        
        deltaTempX = oldDist - stepSize*cos(theta)
        deltaTempY = sin(theta)*stepSize
        newDist = sqrt(pow(deltaTempX,2)+ pow(deltaTempY,2))
        
        # Store new distance value
        self.sensors = newDist
        
    # Reset sensor values
    def reset(self):
        # Move UAV back to starting position from thermal center
        self.sensors = self.planeStartDist
        
    # Return the distance of the UAV from the thermal center
    def distPlane(self):
        return self.sensors