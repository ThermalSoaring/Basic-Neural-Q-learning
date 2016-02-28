# Based on CartPoleEnvironment by Thomas Rueckstiess, ruecksti@in.tum.de

import simpleThermal as simpTh # Contains easy to use Gaussian function

from matplotlib.mlab import rk4
from math import sin, cos, sqrt, pow
import time
from scipy import eye, matrix, random, asarray

from pybrain.rl.environments.environment import Environment


class simpThermEnvironment(Environment):
    """ 
        Provides a simple Gaussian "goodness" function        
    """
    
    # the number of action values the environment accepts
    # The input is cos(theta), where theta is the angle from a line drawn from the plane to the center of the thermal
    # - To illustrate, the input is 1 if the plane travels directly towards the thermal, and 0 if it travels at right angles    
    indim = 1 
    
    # The number of sensor values the environment produces
    # Distance to center of goodness function is provided (where is reward provided?)
    outdim = 1

    # We set the distance of the plane from the center of the thermal randomly
    randomInitialization = True

    def __init__(self, maxPlaneStartDist, stepSize):
        # distPlaneRange specifies the maximum distance the plane can be from the center on startup
        self.maxPlaneStartDist = maxPlaneStartDist
        
        # stepSize is how far the plan moves each time
        self.stepSize = stepSize
                
        # initialize the environment (randomly)
        self.reset()
        self.action = 0.0
        self.delay = False

    def getSensors(self):
        """ Returns goodness of location after action is carried out.
        """
        return asarray(self.sensors)

    # Performs a provided action
    # The action is theta, where theta is the angle (in radians) from a line drawn from the plane to the center of the thermal
    # - To illustrate, the input is 0 if the plane travels directly towards the thermal, and pi/2 if it travels at a right angle to the thermal center    
    def performAction(self, action):
        self.action = action # This updates theta (angle to move on)
        self.step()

    # Update sensor values (update value of goodness after plane has moved)
    # Uses the current values of self.action
    def step(self):
        # Determine the new distance from the center
        oldDist = self.sensors
        theta = self.action
        stepSize = self.stepSize
        
        deltaTempX = oldDist - stepSize*cos(theta)
        deltaTempY = sin(theta)*stepSize
        newDist = sqrt(pow(deltaTempX,2)+ pow(deltaTempY,2))
        self.sensors = newDist
        
        # Reset is called when environment is constructed
    def reset(self):
        """ re-initializes the environment, setting the plane back at a random distance from the center of the thermal
        """
        if self.randomInitialization:
            planeDist = random.uniform(0, self.maxPlaneStartDist) # The distance the plane is from the center of the thermal
        else:
            planeDist = self.maxPlaneStartDist
            
        # Initialize sensors
        self.sensors = planeDist
        
    # Returns the distance of the plane from the center of goodness
    def distPlane(self):
        return self.sensors

