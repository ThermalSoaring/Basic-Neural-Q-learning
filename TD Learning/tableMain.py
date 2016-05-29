'''
Uses a table-based SARSA method to learn how to move towards the thermal center.
By "table-based" I mean the environment state-space is discretized
--We store Q value estimates for each discretized state
--That is, we store how good we think each action is in that state

Note that SARSA learns the Q values corresponding to the policy it learns under, not the greedy policy
(this is different from Q learning, see here: http://stackoverflow.com/questions/6848828/reinforcement-learning-differences-between-qlearning-and-sarsatd)
'''

# tbm contains some duplicate code from the neural net Q learning method
# This duplication was carried out to allow for testing each method individually
# Ideally this duplication would be removed
import tabledBasedMethods as tbm

# Use a lookup table to store state action values
def tableMain():    
    maxPlaneStartDist = 8   # Starting plane distance from thermal, or the maximum such distance if random placement of plane is allowed
    numAngs = 2             # The number of directions in which the plane is allowed to move
    numDist = 6             # Discretizing distances from center into this many chunks
    thermRadius = 3;        # Standard deviation of thermal
    stepSize = 0.1          # How far the plane moves on each interaction     
    
    tbm.tableBasedMethod(maxPlaneStartDist,numAngs,numDist,thermRadius,stepSize)
    
if __name__ == "__main__":  
    tableMain()