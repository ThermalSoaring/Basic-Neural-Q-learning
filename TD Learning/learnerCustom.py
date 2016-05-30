''' 
A modified version of LearningAgent (from the PyBrain library)
-For handling modules with multiple outputs (ex. choose action with highest value output from neural network)
-We are told how good each of the output values is, and we choose the best one
The original code only supports self.module.activate returning a single value, an action
'''
from pybrain.rl.agents import LearningAgent
from pybrain.rl.agents.logging import LoggingAgent   
from numpy import argmax, size    
 
class CustLearningAgent(LearningAgent):

    def getAction(self):
        LoggingAgent.getAction(self)     
           
        # Get the output of the neural network (or table?)
        # --If neural network, returns the value of each of the different actions        
        tempAction =  self.module.activate(self.lastobs)
        if (tempAction.size > 1):     
            bestAction = argmax(tempAction) # Choose the action with the highest value
            self.lastaction = [bestAction]
           
        else: # Original Code (used if the module can directly return the desired action)
            self.lastaction = self.module.activate(self.lastobs)
         
        if self.learning:            
            self.lastaction = self.learner.explore(self.lastobs, self.lastaction)

        return self.lastaction