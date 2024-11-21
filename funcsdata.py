#funcsdata.py

from analyze import *
from loop import *
from objects import *
from simulate import *

def varplotalldata(a):
    varover=[]
    #varstd=[]
    for i in range(a.k+1):
        varover.append(varpltover(i, a))
        #quarter = np.array_split(sorted(a.pmat(i), key=lambda k: random.random()), 4)
        #qtrvar = []
        #for j in range(4):
            #calculate the variance across all quarters
            #qtrvar.append(np.var(quarter[j]))
        #calculate the standard deviation of variance across quarters 
        #varstd.append(np.std(qtrvar))
    result=(2**(np.arange(a.k+1)), varover)
    print('variance scaling analysis complete')
    return result

def eigplotalldata(a):
    eigs=[]
    xplot=[]
    for i in np.arange(4,a.k+1):
        plot=eigpltdata(i,a)
        plot[np.where(plot < 1.*10**(-7))]=0.
        xplot.append(np.arange(1,plot.size+1)/(plot.size))
        eigs.append(plot)
    #result=np.array((xplot,eigs))
    result=np.array((xplot,eigs), dtype=object)
    print('eigenvalue spectra analysis complete')
    return result

def eigpltdata(i, a):
    """
   Same as above except slightly different for imported data.
    
    """
    hold=[]
    for j in range(int(a.N/2**i)):
        hold.append(a.eigsnewdata(i, j)[0])
    hold=np.mean(np.vstack(hold), axis=0)
    return hold

class databoot:
    def __init__(self, rate, ratex, coeff, coeffx,\
    eigspec, eigspecx, var, varx, psil, psilx, \
            mu, muerr,alpha, alphaerr,beta, betaerr,N, area, actmom, actmomx, kurtosis):
        self.rate=rate
        self.ratex=ratex
        self.coeff=coeff
        self.coeffx = coeffx
        self.eigspec=eigspec
        self.eigspecx = eigspecx
        self.var=var
        self.varx = varx
        #self.varerr = varerr
        self.psil=psil
        self.psilx = psilx
        self.mu=mu
        self.muerr=muerr
        self.alpha=alpha
        self.alphaerr=alphaerr
        self.beta=beta
        self.betaerr=betaerr
        self.N =N
        self.area = str(area)
        self.actmom = actmom
        self.actmomx = actmomx
        self.kurtosis = kurtosis

class dataset:
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
        self.clusterlist=selectstep(calcrg\
            (self.pmat,self.cluster, self.k)[1], self.k) 
        # array holding cell indices for each cluster at each RG step
        # stuff only needed for momentum space RG
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
    
    def clusterstepdata(self,i): 
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
    def clustertraindata(self, i, j): 
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
        return self.pmat[self.clusterstepdata(i-1)[j, :],:]
    
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

    def covnewdata(self, i, j): 
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
        data= self.clustertraindata(i,j)
        data -= np.reshape(data.mean(axis=1), (data.shape[0], 1))
        return np.cov(data)
        
    def eigsnewdata(self, i, j): 
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
        return eiggen(self.covnewdata( i , j))

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

class dataload:
    def init(dataset, dataname, clusters):
        self.npu = np.load(dataset)
        self.area = dataname
        self.K = K
        #adding
        self.N=len(self.npu)
        a = dataset(self.N, pmat, K)
        #add kurtosis stuff later
