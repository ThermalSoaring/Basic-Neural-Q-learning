# Based on mdp task by Thomas Rueckstiess, ruecksti@in.tum.de

from pybrain.rl.environments import Task
from scipy import array, asarray
from math import pi, sqrt, exp

# This task is designed to interact with a simpThermEnvironment environment
# When this task is created, a simpThermEnvironment environment is attached to it
# --We can refer to this environment by self.env

class SimpThermTask(Task):

    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        # Give a reward corresponding to how close to the thermal we are
        # The reward follows  a normal distribution with the standard deviation of the thermal
        distToThermal = self.env.sensors         
        sigma = self.env.thermRadius
        reward = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal,2)/(2*pow(sigma,2)))   
        
        return reward        

    def performAction(self, action):
        """ The action vector is stripped and the only element is cast to integer and given
            to the super class.
        """
        Task.performAction(self, int(action[0]))


    def getObservation(self):
        """ The agent returns its distance to the center of the thermal.
        This information is acquired through the environment.
        """
        distToThermal = self.env.getSensors() 
        obs = distToThermal
        
        return obs
    
    # Allows for direct access of the float distance to the center of the thermal
    def getDist(self):
        dist = self.env.sensors
        return dist



