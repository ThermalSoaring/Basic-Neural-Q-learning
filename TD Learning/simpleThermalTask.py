# Forms an interface between the agent (UAV) and the environment (thermal)
# Based on mdp task by Thomas Rueckstiess, ruecksti@in.tum.de
# Interacts with a simpThermEnvironment or contThermalEnvironment environment

from pybrain.rl.environments import Task
from scipy import array, asarray
from math import pi, sqrt, exp

class SimpThermTask(Task):
    # When this task is created, an environment is attached to it (self.env)

    # Get reward based on current sensor values
    def getReward(self):
        
        # Give a reward corresponding to the boost we're getting from the thermal
        # Currently assuming a Gaussian shaped thermal
        distToThermal = self.env.sensors         
        sigma = self.env.thermRadius
        reward = 1/(sigma*sqrt(2*pi)) * exp(-pow(distToThermal,2)/(2*pow(sigma,2)))   
        
        return reward        

    # Hands action to super class
    def performAction(self, action):
        Task.performAction(self, int(action[0]))

    # The agent returns its distance to the center of the thermal.
    # This information is acquired through the environment.
    def getObservation(self):        
        distToThermal = self.env.getSensors() 
        obs = distToThermal
        
        return obs
    
    # Allows for direct access of the distance to the center of the thermal (not discretized)
    def getDist(self):
        dist = self.env.sensors
        return dist