#This is everything needed to run the various RG analyses
#this one includes the infoset class

import numpy as np
from simulate import *
from loop import *
from objects import *
import copy


#infoset class, which is a in simulation
class infoset:
    def __init__(self, N, pmat, k):
        """
        This object, named a in the simulation, is responsible for holding 
        the necessary results and attributes of the RG analysis.
        ----------------------------------------------------------------------
        Inputs: 
        N: number of cells in network
        pmat: activity array holding spike trains for all cells. 
              shape (N, loop*xmax*dt)
        k: number of RG steps taken
        """
        self.N=N # number of cells
        self.k=k # number of RG steps taken
        self.pmat=pmat 
        # activity array holding spike trains for all cells. 
        # shape (N, loop*xmax*dt)
        self.cluster=np.reshape(np.arange(self.N), (np.arange(self.N).size,1)) 
        # empty cluster array holding 0th rg iteration's indices

        #select step in fourth.py
        self.clusterlist=selectstep(calcrg\
            (self.pmat,self.cluster, self.k)[1], self.k) 
        # array holding cell indices for each cluster at each RG step
        # stuff only neede for momentum space RG
        self.flucs=fluc(self.pmat) 
        # array holding each cells' fluctuation from mean firing rate 
        self.eigvector=eigmom(self.pmat) 
        # array holding all covariance array's eigenvectors
    
    def pmatarr(self,i): 
        """
        returns array holding activity of all clusters at given RG step i
        --------------------------------------------------------
        Inputs:
        i: desired RG step at which we want activity of all clusters
        -------------------------------------------------------
        Output: array of shape (number of clusters, xmax*dt*loops) 
                in which the number of clusters is N/2**k
        """
        return calcrg(self.pmat,self.cluster, i)[0]
    
    def clusterstep(self,i): 
        """
        returns the list of indices in each cluster at given RG step i
        ---------------------------------------------------------------------------------
        Inputs:
        i: desired RG step at which we want activity of all clusters
        -------------------------------------------------------------------------------
        Output: array of shape (number of clusters, number of cells in each cluster)
                in which the number of clusters is N/2**i and the number of cells 
                in cluster is 2**i   
        """
        return self.clusterlist[self.k-i] 
        # [k-i], because recall that clusterlist is in reverse order of RG step
    def clustertrain(self, i, j): 
        """
        returns spike trains for cells within a given cluster j at RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        --------------------------------------------------------
        Output: array of shape (number of cells in cluster, xmax*loops*dt) 
                in which the number of cells in cluster is 2**i       
        """
        #print(self.pmat.shape)
        return self.pmat[self.clusterstep(i)[j, :],:]
    
    def corrnew(self, i, j): 
        """
        returns correlation matrix for members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        ----------------------------------------------------------
        Output: array of shape (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in which the 
                number of cells in cluster is 2**i    
        """
        data= self.clustertrain(i,j)
        data -= np.reshape(data.mean(axis=1), (data.shape[0], 1))
        return np.corrcoef(data)

    def covnew(self, i, j): 
        """
        returns covariance matrix for members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        ----------------------------------------------------------
        Output: array of shape (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in which the 
                number of cells in cluster is 2**i    
        """
        data= self.clustertrain(i,j)
        data -= np.reshape(data.mean(axis=1), (data.shape[0], 1))
        return np.cov(data)
        
    def eigsnew(self, i, j): 
        """
        returns sorted eigenvalues for covariance matrix for 
        members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        -----------------------------------------------------------
        Output: eigenvalues of the covariance matrix of shape 
                (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in 
                which the number of cells in cluster is 2**i   
        """
        return eiggen(self.covnew( i , j))
        
    def eigsnewcorr(self, i, j): 
        """
        returns sorted eigenvalues for correlation matrix for 
        members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        -----------------------------------------------------------
        Output: eigenvalues of the correlation matrix of shape 
                (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in 
                which the number of cells in cluster is 2**i   
        """
        return eiggen(self.corrnew( i , j))

    def varnewover(self, i):    
        """
        returns variance of activity over all clusters at RG step i
        Inputs:
        i: desired RG step at which we want activity of all clusters
        -------------------------------------------------
        Output: variance of over all coarse grained variables at RG step i    
        """
        result=[]
        for j in range(int(self.N/(2**i))):
            result.append(np.var(self.clustertrain(i,j)))
        result=np.mean(np.array(result))
        return result

    def probgen(self, i,j):
        """
        For cluster j, calculate probability that every cell in cluster is silent 
        --------------------------------------------------------------------------------------
        Inputs:
        i: desired RG step 
        j: desired cluster index
        --------------------------------------------------------
        Output: probability that cluster j is silent at RG step i
        """

        pmat=self.clustertrain(i, j)  # returns spike trains for a given cluster j at RG step i
        calc=(1.-np.mean(pmat)) # calculate probability of silence within cluster
        return calc

def selectstep(clusters, k):
    """
    Returns the indices of cells in each cluster at each RG step
    -----------------------------------------------------------
    Inputs:
    clusters: the resulting cluster array of the last (kth) RG step.
              shape: (N/(2**k), 2**k)
    k: total number of RG steps performed. shape: scalar
    -----------------------------------------------------------
    Output: array holding the cell indices in each cluster at each RG step.   
            Note in reverse order:
            first subarray is the last RG step, last subarray is the 0th RG 
            step
            shape: holds k arrays, each (N/(2**i), 2**i)
    """
    clusterlist=[]
    for i in 2**np.arange(k):
        clusterlist.append(np.vstack((np.hsplit(clusters, i)))) # split up 
        # into the clusters added at each step
    clusterlist.append(clusters.flatten().T) # append the original 
    # "clusters" (1 cell)
    return clusterlist

def fluc(pmat):
    """
    Calculate fluctuations in preparation for projection onto chosen 
    eigenvectors for momentum space RG
    -----------------------------------------------------------
    Inputs:
    pmat: activity matrix holding all cells' spike trains
    ------------------------------------------------------------
    Output: array holding fluctuations away from mean for each cell
    """
    return pmat - np.reshape(np.mean(pmat, axis=1), (pmat.shape[0],1)) 

def eigmom(pmat):
    """
    Calculate the eigenvectors in preperation for momentum space RG
    -----------------------------------------------------
    Inputs:
    pmat: the activity array of all cells' spike trains. 
          shape: (N, xmax*dt*loop)
    -------------------------------------------------------------
    Output: array of eigenvectors. Each eigenvector is a column in this array. 
            shape: (N,N)
    """
    corr=np.cov(pmat)
    #np.fill_diagonal(corr, 0.)
    corr[np.where(np.isnan(corr)==True)]=0.
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    #print(eigs[0].shape)
    # activity
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    return eigvec

def calcrg(pmatnew, clusternew, k):
    """
    Perform real space RG step using RGrealstep(pmat, cluster, corr)
    --------------------------------------------------------
    Inputs:
    pmatnew: array holding all cells' spike trains. shape: (N, xmax*dt*loop)
    clusternew: array holding the cells which make up each cluster. 
    shape: (N,1)
    k: number of RG steps to be performed. shape: scalar
    -------------------------------------------------------
    Output: spike train for cell i. shape: (N/(2**i), xmax*dt*loop)
            updated cluster array. shape: (N/(2**i), 2**i)
    
    """
    corr=np.corrcoef(pmatnew) #calculate correlation matrix  
    for i in range(0, k): # for every RG step
        pmatnew, clusternew = RGrealstep(pmatnew, clusternew, corr) # perform 
        # RG step
        corr=np.corrcoef(pmatnew) # calculate new correlation matrix
    return pmatnew, clusternew

def RGrealstep(pmat, cluster, corr):
    """
    Perform real space RG step
    Here we first calculate the correlation matrix, c_(ij)= (C_(ij))/
   (sqrt(C_(ii)*C_(jj)))
    Where C is the covariance matrix
    A complication here is that if a cell i never fires or always fires, 
    C_(ii)=0. 
    Thus c_(ij) will be undefined.
    To deal with this I set Nans in the covariance matrix to 0. 
    
    I then set the diagonal to Nan so we do not count cells twice. Then pick 
    out maximally correlated
    cells and combine their activities, set the cell's corresponding rows and 
    columns to Nan.
    Then iterate until entire array is Nan.
    Update clusters at every iteration
    ---------------------------------------------------------
    Inputs:
    pmat: array holding all cells' spike trains. shape:(N, xmax*dt*loop)
    cluster: array holding the cells which make up each cluster. shape: (N,1)
    corr: correlation matrix, note that this may have Nans in it
    ----------------------------------------------------
    Output: RG transformed activity array. shape: (N/(2**i), xmax*dt*loop), 
            updated cluster array. shape: (N/(2**i), 2**i)
    
    """
    j=0
    corr1=copy.deepcopy(corr) # make a copy of correlations for processing
    corr1[np.where(np.isnan(corr1)==True)]=0. # set Nans to 0
    np.fill_diagonal(corr1, None) # set diagonal to Nan so we dont double 
    # count cells
    pmat1=copy.deepcopy(pmat) #make a copy of spike trains for processing
    pmatnew=np.zeros((int(pmat1.shape[0]/2), pmat1.shape[1])) #holds post RG 
    # step activity
    clusternew=np.zeros((int(pmat1.shape[0]/2), 2*cluster.shape[1])) #holds 
    # post RG clusters
    while j != pmatnew.shape[0]: #while new activity array is not filled up
        maxp=np.nanmax(corr1.flatten()) #pick out maximum non-Nan correlation
        wh=np.array(np.where(corr1==maxp)) #pick out indices where max corr is 
        # present
        i=np.random.choice(np.arange(wh.shape[1])) #choose the random index of 
        # these indices
        #i=np.min(np.where(np.abs(wh[1]-j)==np.min(np.abs(wh[1]-j))))
        #now we have 2 maximally correlated cells, wh[0,i] and wh[1,i]
        #now set rows and columns corresponding to these cells to Nan
        corr1[wh[0, i], :]=None
        corr1[:, wh[0,i]]=None
        corr1[wh[1, i], :]=None
        corr1[:, wh[1, i]]=None
        #now add activities of our chosen cells wh[0,i] and wh[1,i] and update 
        # clusters
        calc=pmat1[wh[0, i], :]+pmat1[wh[1, i], :]
        pmatnew[j, :]=calc
        clusternew[j, :]= np.concatenate(np.array([cluster[wh[0,i], :],\
            cluster[wh[1,i], :]]), axis=None)
        j += 1 #we have completed a row of the new activity array, count it!
        if j== pmatnew.shape[0]: #break if we have completed counting
            break 
    return pmatnew.astype(int), clusternew.astype(int)

def eiggen(corr):
    """
    Calculate eigenvalues and sort largest to smallest
    ---------------------------------------------------------
    Inputs:
    corr: input correlation matrix. shape: (number of cells, number of cells)
    ----------------------------------------------------------
    Output: sorted eigenvalues for correlation matrix corr. 
            shape: (number of cells,)
    """
    
    eigs=np.linalg.eig(corr)
    arg=np.argsort(eigs[0])[::-1]
    eigvals=eigs[0][arg]
    eigvecs=eigs[1][:,arg]
    return eigvals, eigvecs
    
def normmom(ppmat):
    """
    Makes the sum of squares of momentum space RG'd activity equal to 1
    ----------------------------------------------------------------------------
    Inputs: 
    ppmat: momentum space RG'd activity array.  shape: (N/l, xmax*dt*loop)
    --------------------------------------------------------------------------
    Output: normalized momentum space RG'd activity array.  shape: (N/l, xmax*dt*loop)
    """
    ppmatnew=np.empty(ppmat.shape)
    for i in range(ppmat.shape[0]): #enforce that sum of squares must be 1
        test=(np.sqrt(ppmat.shape[1])*ppmat[i,:])/(np.sqrt(np.sum(ppmat[i,:]**2)))
        vartest=np.mean(test**2)
        ppmatnew[i,:]=test
    return ppmatnew

def RGmom(l, a):
    """
    Perform momentum space RG step
    --------------------------------------------------------------------------------------
    Inputs:
    l: total number of eigenvectors/l = number of eigenvectors I will 
        project fluctuations onto. shape:scalar
    a: object
    --------------------------------------------------------------------------------------
    Output: RG transformed activity array. shape: (N/l, xmax*dt*loop)
    """
    eigvec=a.eigvector[:,:int(a.eigvector.shape[1]/l)] #sort eigenvectors, cut out some
    ppmat=np.dot(eigvec,np.dot(eigvec.T,a.flucs))
    #print(ppmat.shape)
    #project fluctuations onto chosen eigenvectors
    return ppmat

def autocorr(series, norm='real'):

    """
    generate normalized autocorrelation function
    ---------------------------------------------------
    Inputs:
    series: 1D array holding sequence I wish to calculate normalized 
            autocorrelation function of
    norm: if 'real': returns autocorrelation normalized by variance (default)
          if 'mom': returns autocorrelation normalized by mean fluctuations 
          squared, appropriate for momentum space
    --------------------------------------------------
    Output:
    normalized correlation function, of shape series.size+2
    """
    
    plotcorr = np.correlate(series,series,'full')
    nx = int(plotcorr.size/2)
    lags = np.arange(-nx, nx+1) # so last value is nx
    plotcorr /= (len(series)-lags)
    plotcorr -= np.mean(series)**2
    if norm == 'real':
        plotcorr /= np.var(series)
    if norm == 'mom':
        plotcorr /= np.mean((series-np.mean(series))**2)
    return lags, plotcorr