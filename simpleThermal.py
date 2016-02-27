import numpy as np

# Returns distance between (x,y) and mu
def dist(x,y,mu):
    return np.sqrt( np.power(x-mu[0],2) + np.power(y-mu[1],2))

# Returns a value on a 2D normal distribution
# R is the characteristic thermal radius
# W is the characteristic thermal strength (maximum value)
def norm2D(x,y,mu,W,R):
    return W*np.exp(-np.power(dist(x,y,mu)/R,2))