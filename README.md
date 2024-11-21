## neurocg

To run this code, use command line. I recommend starting with stimloop.py and stimsweep.py to get a feel for it. 

#stimsweep.py

This runs a set of simulations with varying latent fields you can set. It saves the files as pickled objects in a folder.

#stimloop.py

This applies a series of pRG analyses to the saved objects.

After running these from command line, you can plot various exponents in the ipynb file. The following files are sort of organized. 

#simulate.py

This file has things related to simulating the OU process in spiking format

#analyze.py

Applies a bunch of pRG analyses 

#objects.py 

Includes some more basic or utility functions that are variously called and not aptly labeled analyze or simulate. 

#loop

Includes the necessary functions for globbing together simulations. 

#other notes

In https://github.com/mcmorre has other loops (eta, eps, phi, tau) that you can grab.
