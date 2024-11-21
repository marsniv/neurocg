#first layer
#includes everything for simulating the data with the given parameters
import numpy as np 
from loop import *
from objects import *
from analyze import *

#this is the first function called to simulate. leads to environ
def runsim(N0, nstim, time, eta, epsilon, k, inputlabel='normal'):
    N0 = N0 # number of cells
    N = N0
    loop = 200 # number of track runs

    xmax = 50 # track length

    dt=1. # increment for measurement locations

    # set up network structure

    nstim = nstim # number of latent stimuli

    percell= 1 # probability that each field is accepted

    latprob=np.array([1-percell, percell]) 

    vj = 0. # mean of couplings for nonplace fields
  
    sj = 1. # standard deviation of couplings for nonplace fields


    time=time

    if type(time)!=float:
        print('running with non-uniform time constants')
        timeconst=time
        timelabel=inputlabel
        
    else: 
        timeconst = np.full((nstim,), time) # mean length of stochastic process in track lengths
        timelabel=time

    phi=1 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = eta # multiply by this constant to increase overall activity of 
    # network

    epsilon= epsilon

    env = environ(N0=N0, N=N, loop=loop, xmax=xmax, dt=dt, nstim=nstim,\
    percell=percell, \
    latprob=latprob, vj=vj, sj=sj,\
    timeconst=timeconst,\
    phi=phi, eta=eta, epsilon=epsilon)


    k=k
    a=infoset(N0, env.pmat, k)


    name_env='/users/mli853/Documents/placerg-main/data/env_stim{}N{}e{}et{}.pkl'.format(nstim, N, np.round(epsilon, 1), np.round(eta,1))
    
    name_a = '/users/mli853/Documents/placerg-main/data/a_stim{}N{}e{}et{}.pkl'.format(nstim,N, np.round(epsilon,1), np.round(eta,1))

    save_object(env, name_env)
    save_object(a, name_a)

#this environ class simulates the OU process, hamiltonian, and spikes. 
class environ:    
    #defines all the parameters we are interested in
    def __init__(self, N0, N, loop, xmax, dt, nstim, percell,\
                 latprob, vj, sj, \
                 timeconst, phi, eta, epsilon):
        
        #not sure what this does 
        self.choice=np.array([0,1])
        
        #number of neurons
        self.N0=N0 # number of cells
        self.N=N # number of cells after removing silent ones
        
        #i believe the next three have to do with time constant stuff
        self.loop=loop # number of loops run
        self.xmax=xmax # maximum track length
        self.dt=dt # time step

        #number of latent variables (ooo) 
        self.nstim=nstim # number of latent fields

        #number of latent fields per cell
        #how does this diverge from number of latent fields? 
        self.percell=percell # average number of latent 
                                # stimuli assigned per cell
        #stuff to get rid of (eventually...)
        '''
        self.bothprob=bothprob # probability that cell 
                                # is coupled to nonplace field. 
                                # [p(not coupled), p(coupled)] 
        self.placeprob=placeprob # probability that place cell is coupled 
                                # only to a place field. 
                                # [p(only coupled to a place field), 
                                # p(may be coupled to some 
                                # nonplace fields)]
        '''
        #i think we just want to keep the bellow.
        self.latprob=latprob # probability that place cell 
                                # is coupled 
                                # only to a place field. 
                                # [p(only coupled to a place field), 
                                # p(may be coupled to some 
                                # nonplace fields)]

        #vj should be zero i think? 
        self.vj=vj # mean of the latent cell couplings. scalar 
        #should go away
        '''
        self.vjplace=vjplace # mean of the place cell couplings. scalar 
        '''
        #this should be 1 I think
        self.sj=sj #standard deviation of the 
                                # nonplace cell couplings. scalar 
        #go away eventually...
        '''
        self.sjplace=sjplace #standard deviation of the place 
                                # cell couplings. scalar 
        '''
        #time constant stuff
        self.timeconst=timeconst # mean length of stochastic process 
                                # in track lengths
        
        #morrell parameters
        self.phi=phi

        self.eta=eta # multiply by this constant 
                            # to increase overall activity of network
        self.epsilon=epsilon


        #place field stuff
        ''''
        
        self.x = np.tile(np.arange(0,self.xmax, self.dt), (self.N0,1)) 
                            # stack N copies of measurement locations
        self.v=np.random.uniform(0, self.xmax, (1,self.N0)) 
                            # means of place fields. shape: (1,N)
        self.vdev=gamma(self.xmax/10., (self.xmax/20.), self.N0).T 
                            # variances of place fields. shape: (1,N)
        '''
        #latent filed stuff
        self.sigmas= np.full((self.nstim,), 1.) 
                            # standard deviation of stochastic process. 
                            # shape: (nstim,)
        self.taus= self.timeconst*self.xmax
                            #gamma(self.timeconst*self.xmax, \
                            #self.timeconststd*self.xmax, self.nstim).flatten() 
                            # time constant of stochastic process. 
                            # shape: (nstim,)
        self.vs=np.full((self.nstim,), 0.) 
                            # mean of the stochastic process. 
                            # shape: (nstim,)

        #okay our first new function! where will it lead us? 
        #pulled from funcs.py, its related to OU process. put in 
        self.process= stim(self.taus, self.sigmas, self.dt, \
                    int(self.loop*self.xmax*self.dt)).T 
                            # this makes an array of shape 
                            # (loop*dt*xmax, nstim)
        #pretty simple, just creating the OU process over all the time steps in the simulation


        # our second new function. found in funcs.py, put in second.py
        #creating couplings
        self.J=fillJ(np.zeros((self.N0+self.nstim, self.N0)), self.N0,\
                self.vj,\
                self.sj, self.nstim, self.latprob,\
                self.choice, self.phi) 
                            # coupling array. shape (N+nstim, N)

        #third new function, found in funcs.py, copied into second.py
        self.fields=fillfields(self.N0, \
                np.zeros((self.loop, self.xmax, self.N0+self.nstim)),\
                self.process, self.loop) 
                            # fields array. shape (loop, xmax, N+nstim)
        
        #computeh in second.py
        self.h= computeh(self.fields, self.J, self.eta, self.epsilon) # hamiltonian array. 
                            # shape (loop, xmax, N)

        #in second.py
        self.P= computeP(self.h) #  P(silence) array. 
                            # shape: (loop, xmax, N)
        #self.pmatprocess0=_dice6.dice6(self.P) # unreshaped activity 
                            # array, has shape of P: 
                            # shape: (loop, xmax, N)
                            # note that this command is the same as the one below,
                            # but using the faster custom function dice6
       
        #spikesbetter in second.py
        self.pmatprocess=spikesbetter(self.P) # unreshaped activity 
                            # array, has shape of P: 
                            # shape: (loop, xmax, N)
        
        self.pmat=np.vstack(self.pmatprocess).T

        #
        #self.inds=nonzerocell(self.pmat0, self.N)
        self.boots=bootcell(self.pmat, self.N)
        #self.pmat=self.pmat0[self.inds,:] # reshape activity array, 
                            # shape (N, loop*xmax*dt)
        #self.J=np.vstack((self.J0[self.inds,:][:,self.inds], self.J0[self.N0:, self.inds]))
        #self.h=self.h[:,:,self.inds]
        #self.P=self.P[:,:,self.inds]
        #self.p=np.vstack(self.P).T # make (N,loop*dt*xmax) array of 
                            # probabilties
        #self.fields=np.dstack((self.fields[:,:,self.inds], self.fields[:,:,self.N0:]))

        self.counts=np.array((np.count_nonzero(self.J[self.N:,:], axis=0))) 
                            # count number of nonzero entries in 
                            # J[N:,:] for latent cells. 
                            # Returns array 
                            # [(number of cells with latent fields)] 

        '''
        self.placecell=np.array(\
                np.where(np.logical_and(self.counts[0,:] != 0.,\
                self.counts[1,:] == 0.))).flatten()
                                                                                     
                            # holds J indices for cells with 
                            # with only place fields
        self.bothcell= np.array(np.where(np.logical_and(self.counts[0,:] \
                != 0., self.counts[1,:] != 0.))).flatten() 
                                                                             
                            # holds J indices for             
                            # cells with both place and nonplace 
                            # fields
        '''
        #self.latentcell=np.array(\
                #np.where(np.logical_and(self.counts[0,:] == 0.))).flatten()
                                                                                        
                            # holds J indices for cells
                            # with only nonplace fields
        self.latentcell=self.counts
        
#second layer deep.
#these are various functions necessary to run environ class. 

#pulled from funcs.py, used for environ in first.py
def stim(taus, sigmas, dt, leng):
    
    """
    Refer to http://th.if.uj.edu.pl/~gudowska/dydaktyka/Lindner_stochastic.pdf
    Ornstein-Uhlenbeck process, using Euler-Maruyama method.
    Here the mean of the process generated is 0.
    -------------------------------------------------------------
    Inputs:
    recall that nstim is the number of latent stimuli
    sigmas: standard deviation of stochastic process. shape: (nstim,)
    taus: time constant of stochastic process. shape: (nstim,)
    vs: mean of the stochastic process. shape: (nstim,)
    dt: time step. shape: scalar
    leng: desired length of process. in this case the desired length will be 
          loop*dt*xmax
    ----------------------------------------------------------
    Output: states of given latent fields, over time period leng at 
            intervals of dt.
            shape: (nstim, leng) --> (number of latent stimuli, loop*dt*xmax)
    """
    
    #numstim=taus.size
    numstim=len(taus)
    
    gamm=1./taus
    
    D=(sigmas**2)/taus
    
    arr=np.zeros((numstim,leng))
    
    for i in range(1,leng):
        
        rands=np.random.randn(numstim)
        
        arr[:,i]=arr[:,i-1]*(1-gamm*dt)+np.sqrt(2*D*dt)*rands
        
    return arr

#pulled from funcs.py, used for environ in first.py
def fillJ(J, N, vj, sj, nstim, latprob, choice, phi):    
    """
    fill empty J array with entries
    ---------------------------------------------------
    Inputs:
    J: empty J array
    N: number of cells
    vjplace: mean of the place cell couplings. scalar 
    sjplace: standard deviation of the place cell couplings. scalar 
    vj: mean of the latent cell couplings. scalar 
    sj: standard deviation of the latent cell couplings. scalar 
    placeprob: probability that cell is coupled to place field. [p(not 
               coupled), p(coupled)]
    stimprob: probability that cell is coupled to nonplace field. [p(not 
              coupled), p(coupled)] 
    placeonlyprob: probability that place cell is coupled only to a place 
                   field. [p(only coupled to a place field), p(may be coupled 
                   to some nonplace fields)]
    choice: possible spin values. [0,1]
    const: normalize latent part of hamiltonian by this constant. 
           that is: I multiply every latent coupling my const such that when 
           I compile the hamiltonian, I get H(cell) = J_{cell}^{(place)}
           *h_{cell}^{(place)}
                                                                                                                         
           +(1/sqrt(percell)) *\sum_i{(J_{cell, i}^{(nonplace)}*h_{cell, i}
           ^{(nonplace)})

    --------------------------------------------------
    Output:
    J array filled with couplings
    """
    # fill couplings array with entries
    # recall that choice is [0,1]

    if nstim !=0 : 
        wl = np.array(np.where(J[(np.diag_indices(J[:N,:].\
           shape[1]))] ==0)).flatten()
        J[N:,wl] = np.random.normal(vj, sj, \
            J[N:, wl].shape)* np.random.choice(choice, J[N:,wl].shape, \
            p=latprob)
        countcells=np.array((np.count_nonzero(J[N:,wl],axis=0)))
        percell=(phi/np.sqrt(np.mean(countcells)))
        J[N:,wl] *= percell

    return J

#third and final 
def fillfields(N, fields, process, loop):
    """
    fill empty fields array with entries. Note that fields array should be an 
    array of zeros
    ---------------------------------------------------
    Inputs:
    N: number of cells
    x: stack of N copies of measurement locations: as in 
       np.tile(np.arange(0,xmax, dt), (N,1))
    v: means of place cell waveforms. has shape (1,N)
    vdev: standard deviations of place cell waveforms. has shape (1,N)
    fields: empty fields array. has shape (loop, xmax, N+nstim), must be array 
            of zeros
    process: array holding all nonplace fields at every time step. has shape 
             (loop*xmax*dt, nstim)
    loop: number of track runs. integer
    --------------------------------------------------
    Output:
    fields array filled with fields
    """
    # fill fields array with latent stimuli
    #(taus, sigmas, dt, leng)

    fields[:,:,N:] = np.array(np.vsplit(process, loop)) # save filled latent 
    # fields in fields array
    '''
    fields[:,:,:N] = fillplace(fields[:,:,:N], x, v, vdev) # fill fields with 
    # place fields
    '''
    return fields

def computeh(fields, J, eta, epsilon):
    """
    Fast computation of hamiltonian. Uses blis.py matrix multiplication.
    Note that here the maximum field value is subtracted off the hamiltonian
    ---------------------------------------------------
    Inputs:
    fields: fields array. shape (loop, xmax, N+nstim)
    J: coupling array. shape (N+nstim, N)
    --------------------------------------------------
    Output:
    fields array filled with place fields. shape: (loop, xmax, N+nstim)
    """
    h = blis_gemm(fields,J) # perform dot product to make hamiltonian
    # note that this above function using the external blis.py library.
    # it is faster that np.dot
    #h = np.dot(fields,J) # perform dot product to make hamiltonian
    h *= eta
    h += epsilon
    return h

def computeP(h):
    """
    Compute probablilities of silence given hamiltonian array
    ---------------------------------------------------
    Inputs:
    h: hamiltonian array. shape (loop, xmax, N)
    --------------------------------------------------
    Output:
    P(silence) array. shape: (loop, xmax, N)
    """   
    return 1./(1+np.exp(h)) # compute the probability of silence

def spikesbetter(P):
    """
    same as the custom cython function _dice6, a python implementation for easy use on other computers
    does spin selection procedure based on given array of probabilities
    --------------------------------------------------------------------
    Inputs:
    P: probability of silence array. shape (loop, xmax, N)
    -------------------------------------------------------------------
    Output: 
    array of spin values in {0,1} with shape (loop, xmax, N) 
    """
    spikes=np.zeros(P.shape)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for k in range(P.shape[2]):
                if np.random.rand() > P[i,j,k]:
                    spikes[i,j,k] += 1
    return spikes


def bootcell(arr, keeps):    
    num=int(arr.shape[1]/4)
    inds=np.random.randint(low=num, high=arr.shape[1], size=4)
    s=[]
    for i in range(4):
        binds=(nonzerocell(arr[:, inds[i]-num:inds[i]], keeps))
        s.append(arr[binds, inds[i]-num:inds[i]])
    return s

def nonzerocell(pmat, size):
    """
    Remove silent cells from ensemble.
    -----------------------------------------------------
    Inputes:
    pmat: activity array shape shape:(Number of cells, number of time steps)
    ------------------------------------------------------------------
    Output:
    pmatnew: activity array with silent cells removed shape:(Number of cells, number of time steps)
    """
    means=(np.mean(pmat, axis=1))
    wh=np.where(means>0.)[0]
    #print(str(pmat.shape[0]-wh.size) +  ' cells were silent and therefore removed')
    select=np.random.choice(wh.size, size=size, replace=True)
    return wh[select]