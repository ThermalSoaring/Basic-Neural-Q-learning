import sys # Allows us to force printing
import simpleThermal as simpTh # Contains easy to use Gaussian function
import simpleThermalEnvironment as thermEnv # Very basic distance updating environment

import numpy as np
from math import pi

# Plotting libraries:
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Create environment
maxPlaneStartDist = 0.5
stepSize =0.1
env = thermEnv.simpThermEnvironment(maxPlaneStartDist, stepSize)  

print(env.distPlane())

for x in range(0, 5):
    # Move the plane towards the thermal
    theta = pi
    env.performAction(theta)
    
    # Print the new distance to the center
    print(env.distPlane())
'''
# Set thermal parameters
mu = [0,0]
W = 1
R = 2

# Find surface points (we can set mu = mean, W = thermal height, and R = radius)
Z = simpTh.norm2D(X,Y,mu,W,R)
#mlab.bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0)

# Define mesh points for surface
npts = 100
minPlot = -3
maxPlot = 3
x = np.linspace(minPlot, maxPlot, npts)
y = np.linspace(minPlot, maxPlot, npts)
X,Y = np.meshgrid(x,y)

# Create contour plot
plt.contour(X,Y,Z)
plt.show()
'''


