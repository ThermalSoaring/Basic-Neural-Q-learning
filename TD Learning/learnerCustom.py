# A modified version of LearningAgent
# The idea is to make the best choice based on multiple outputs from a neural network
# The original code only supports self.module.activate returning a single value, an action
from pybrain.rl.agents import LearningAgent
from pybrain.rl.agents.logging import LoggingAgent   
    
class CustLearningAgent(LearningAgent):

    def getAction(self):
        """ Activate the module with the last observation, add the exploration from
            the explorer object and store the result as last action. """
        LoggingAgent.getAction(self)       
        
        # Here is where the table or neural network returns the action
        # This consists of the values of the different actions
        # We choose the action with highet value 
       
        from numpy import argmax, size        
        tempAction =  self.module.activate(self.lastobs)
        if (tempAction.size > 1):     
            bestAction = argmax(tempAction)
            self.lastaction = [bestAction]
           
        else: # Original Code (used still for stuff like table lookup)
            self.lastaction = self.module.activate(self.lastobs)
         
        if self.learning:            
            self.lastaction = self.learner.explore(self.lastobs, self.lastaction)

        return self.lastaction
