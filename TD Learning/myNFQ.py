# Edited version (for debugging / various half-tested additional features) of PyBrain's nfq.
# Very much an "in progress" file
# -There are a number of half-tested attempted fixes in this code (some commented out)
from scipy import r_

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import one_to_n

import numpy as np

# Debug printing functions
def printSAValsNet(net, states):
    for state in states:
        print(state, ':\t', np.transpose(net.getActionValues(state)))

def printSADiff(net, states, refState, refAction):
    refVal = net.getActionValues(refState)[refAction]
    for state in states:
        print(state, ':\t', np.transpose(net.getActionValues(state)) - refVal)

def printSAChange(modbefore, modafter,states):
    for state in states:
        print(state, ':\t', np.transpose(modafter.getActionValues(state)- modbefore.getActionValues(state)))
      
        
class NFQ(ValueBasedLearner):
    """ Neuro-fitted Q-learning"""    
    
    def __init__(self, sizeBatch, numBatchToKeep, maxEpochs=20):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.7 # (How farsighted we are. 0 = very shortsighted, 1 = very farsighted)
        self.maxEpochs = maxEpochs
        
        # Added to keep track of more training examples
        self.supervisedMaster = None
        self.sizeBatch = sizeBatch
        self.numBatchToKeep = numBatchToKeep     
        
        # Keep track of how old each sample is
        # The younger the sample, the higher the age
        self.ageList = []
        self.age = 0
        
        # Keep track of where each sample is from
        self.locList = []
        
        # Keep track of the action in each case
        self.actList = []
        
        # Added
        self.learningRateStart = 100
        
    def learn(self):
        # Added:  Create larger training set
        if (self.supervisedMaster == None):
            self.supervisedMaster = SupervisedDataSet(self.module.network.indim, 1)
        
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = SupervisedDataSet(self.module.network.indim, 1) # numInput, numOutputs   
        
        # Add previous examples to the current training data
        for trainingSample in self.supervisedMaster: 
            supervised.addSample(trainingSample[0], trainingSample[1])
        
        # Trim old samples
        # numToKeep = self.sizeBatch * self.numBatchToKeep
        # if (len(supervised) > numToKeep):
            # supervised = SupervisedDataSet(inp=supervised['input'][-numToKeep:].copy(),
                                   # target=supervised['target'][-numToKeep:].copy())            
        
     
        # Modified: makes rewards bigger or smaller for debugging
        rewardScale = 1        
        # End modified      
        
        for seq in self.dataset:
            lastexperience = None
            for state, action, reward in seq:
    
                if not lastexperience:
                    # delay each experience in sequence by one
                    lastexperience = (state, action, reward)
                    continue

                # use experience from last timestep to do Q update
                (state_, action_, reward_) = lastexperience
                
                # Modified
                reward_ = rewardScale * reward_
                # End modified
                
                Q = self.module.getValue(state_, action_[0])  
                
                inp = r_[state_, one_to_n(action_[0], self.module.numActions)]
                
                # Debug
                #print('Q value before:',Q, '(old state:', state_, ', action:', action_[0], ', reward:', reward_,')')
                learnRate = 0.5  #(self.learningRateStart/(self.age+1)) + 0.5 # originally just 0.5
                tgt = Q + learnRate*(reward_ + self.gamma * max(self.module.getActionValues(state)) - Q)
                # print('delQ:',tgt - Q)
                # Debug                
                #print('Q value after:', tgt)
                supervised.addSample(inp, tgt)
                
                # Add training example to master list
                self.supervisedMaster.addSample(inp, tgt)                
                # Record age and location 
                self.ageList.append(self.age)                
                self.locList.append(inp[0])
                self.actList.append(inp[1])
               
                # update last experience with current one
                lastexperience = (state, action, reward)
        
        
        # Remove redundancy 
        # If older sample is within this range of newer sample, 
        # then the older sample is deemed redundant            
        redClose = 0.1
        redAge = 1
        i = 0
       
        superOldRemove = 20
        toRemoveIndexList = []
        for i in range(len(supervised)):
            j = 0
            # If there is another newer sample within some close distance
            # of this one, remove this sample
            for j in range(len(supervised)):  
                if (self.ageList[j] + superOldRemove < self.age): # Remove super old stuff
                    toRemoveIndexList.append(j)
                    continue
                if (abs(self.locList[i] - self.locList[j]) < redClose and self.actList[i] == self.actList[j]): # If close and same action              
                    if (self.ageList[i] + redAge < self.ageList[j]):
                        toRemoveIndexList.append(i)                        
                    elif(self.ageList[j] + redAge < self.ageList[i]):
                        toRemoveIndexList.append(j)                        
                j = j + 1
            i = i + 1     

        # Remove redundant ones
        toRemoveIndexList =   np.unique(toRemoveIndexList)
        toKeepIndexList = [x for x in range(len(supervised)) if x not in toRemoveIndexList]
        supervised = SupervisedDataSet(inp=supervised['input'][toKeepIndexList].copy(),
            target=supervised['target'][toKeepIndexList].copy())     
        
        self.age = self.age + 1
                           
        print(supervised)
        #input()
      
        # train module with backprop/rprop on dataset
        #trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=False)
        #trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)
        

        
        # PROMOTE STABILITY - add more examples
        # stabLocs = np.arange(0,10,0.5)
        # for loc in stabLocs: # Go through all locations we are going to be stable at            
            # for currAction in range(self.module.numActions):      
                # inp = r_[loc, one_to_n(currAction, self.module.numActions)]
                # tgt = self.module.getActionValues(loc)[currAction][0]
                # supervised.addSample(inp,tgt)
           
        
        # print('With additional examples:')       
        # Add additional training examples to prevent network forgetting what it learned        
        # stabStates = np.arange(0,10,0.5)
        # for i in stabStates: # Currently looking at values at 0, 1, 2, 3.., 9 away from center of thermal
            # currState = i
            # from scipy import argmax
            # currAction = argmax(self.module.getActionValues(currState))
            # currVal = self.module.getActionValues(currState)[currAction][0]            
          
            # inp = r_[currState, one_to_n(currAction, self.module.numActions)]
            # tgt = currVal
            # supervised.addSample(inp,tgt)
            
        # alternative: backprop, was not as stable as rprop
    
        # Train
        trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.005, batchlearning=True, verbose=False)         
        trainer.trainUntilConvergence(maxEpochs=50)

        #PRINT POLICY
        print('Policy:')
        for i in range(40): #numDist
            from scipy import argmax
            print(i/4, ':\t', argmax(self.module.getActionValues(i/4)), '\t', self.module.getActionValues(i/4)[0],self.module.getActionValues(i/4)[1])#,self.module.getActionValues(i/4)[2])        
        
        # PRINT CHANGES        
       # printSAValsNet(self.module, np.arange(0,10,0.5)) 
        refState = 0
        refAction = 0
        # printSADiff(self.module, np.arange(0,10,0.5), refState, refAction)
        # print()
        # printSAChange(modCopy, self.module,np.arange(0,10,0.5))
        # print()
        # input()
        
        # End modified