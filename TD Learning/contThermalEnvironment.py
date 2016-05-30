''' 
Based on CartPoleEnvironment by Thomas Rueckstiess, ruecksti@in.tum.de
Provides sensor values and updates sensor values based on actions (uses Gaussian shaped thermal)
Returned sensor values are continuous
-Currently, only sensor value is distance from center of thermal

Used by the neural fitted Q method, not the table based method
'''
from math import sin, cos, sqrt, pow, pi, floor

from pybrain.rl.environments.environment import Environment

class contThermEnvironment(Environment):   
    # The number of dimensions in the action vector
    # The input is cos(theta), where theta is the angle from zero along a line drawn from the plane to the center of the thermal
    # - To illustrate, the input is 1 if the plane travels directly towards the thermal, and 0 if it travels at right angles    
    indim = 1 
    
    # The number of sensor values the environment produces
    # Currently only sensor is distance from center of thermal
    outdim = 1

    def __init__(self, planeStartDist, stepSize, numAngles, thermRadius):
        # planeStartDist is the distance the plane starts from the thermal center
        self.planeStartDist = planeStartDist
        
        # stepSize is how far the UAV moves each time
        self.stepSize = stepSize
                
        # numAngles is the number of discrete directions the plane can move
        self.numAngles = numAngles        
       
        # Sets standard deviation of normal shaped reward function
        self.thermRadius = thermRadius
        
        # Initialize the stored values for the states and actions of the agent
        self.reset()
        self.action = 0.0
        self.delay = False
        
    # Returns (non-discretized) distance to center of thermal
    def getSensors(self):
        return [self.sensors]
 
    # Carries out the action provided by the agent
    # The action is theta, where theta is the angle (in radians) from a line drawn from the plane to the center of the thermal
    # - To illustrate, the input is 0 if the plane travels directly towards the thermal, and pi/2 if it travels at a right angle to the thermal center    
    def performAction(self, action):   
        self.action = action # This updates theta (angle to move on)
        self.step()

    # Update sensor values to capture the movement of the UAV, given the current intended action
    def step(self):
        # Determine the new distance from the center
        oldDist = self.sensors

        # We need to convert the action into radians
        # Assume we have numAngles = n+1
        # Index 0           ->      0 radians
        # Index last (n)    ->      pi radians
        # Index a           ->      a/n*pi radians = a/(numAngles-1)*pi
        
        # Check:  (with three choices)
        # action 0 -> 0 radians
        # action 1 -> 1/(2)*pi        
        theta = self.action/(self.numAngles-1)*pi;
        stepSize = self.stepSize
                        
        deltaTempX = oldDist - stepSize*cos(theta)
        deltaTempY = sin(theta)*stepSize
        newDist = sqrt(pow(deltaTempX,2)+ pow(deltaTempY,2))
        self.sensors = newDist
        
    # Resets sensor values
    def reset(self):
        # Move UAV back to starting position from thermal center
        self.sensors = self.planeStartDist  
        
    # Return the distance of the UAV from the thermal center
    def distPlane(self):
        return self.sensors