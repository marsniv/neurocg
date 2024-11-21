#main.py
# set up track structure

import numpy as np
from simulate import *
from analyze import *
from objects import *
from loop import *
array = np.array([1, 2, 3, 4, 5, 10, 20])
for i in array:
    N0 = 512 # number of cells

    nstim = i # number of nonplace stimuli
        
    time= 0.1

    eta = 2.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= -6.0

    k = 8
    
    runsim(N0, nstim, time, eta, epsilon, k)
    print('sweep ' + str(i) + ' complete')
