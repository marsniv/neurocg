#these are functions needed to run the loop, but also any other short functions that need a home.

import glob
import numpy as np
from scipy.optimize import curve_fit
from simulate import *
from loop import *
from analyze import *
from blis.py import gemm 
import pickle
import copy


def save_object(obj, filename):
    """
    Save python object
    -------------------------------------------
    Inputs:
    obj: object I want to save
    filename: name of pickle file I want to dump into. Example: 'dump_file.pkl'
    """

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """
    Load python object
    -------------------------------------------
    Inputs:
    filename: name of pickle file I want to load. Example: 'dump_file.pkl'
    -------------------------------------------
    Output:
    loaded object
    """
    with open(filename, "rb") as input_file:
        e = pickle.load(input_file)
    return e

class bootstrap:
    def __init__(self, rate, coeff, eigspec, var, psil, actmom, autocorr, tau, mu, alpha, beta, z):
        self.rate=rate
        self.coeff=coeff
        self.eigspec=eigspec
        self.var=var
        self.psil=psil
        self.actmom=actmom
        self.autocorr=autocorr
        self.tau=tau
        self.mu=mu
        self.alpha=alpha
        self.beta=beta
        self.z=z

def cellraterank(env):
    plott=(np.arange(1,env.N+1)/env.N,np.sort(np.mean\
            (env.pmat, axis=1))[::-1])
    return plott

def corrcoefhist(env):
    samp=np.corrcoef(env.pmat)
    shuff=copy.deepcopy(env.pmat)
    shuff=rollx(shuff)
    sampshuff=np.corrcoef(shuff)
    np.fill_diagonal(samp,0.)
    np.fill_diagonal(sampshuff,0.)
    samp=samp.flatten()
    sampshuff=sampshuff.flatten()
    samp[np.where(np.isnan(samp)==True)]=0.
    sampshuff[np.where(np.isnan(sampshuff)==True)]=0.
    pltcorr=drawpdf(samp,100)
    pltcorrshuff=drawpdf(sampshuff,100)
    result=np.array((pltcorr, pltcorrshuff))
    return result

def eigplotall(a):
    eigs=[]
    xplot=[]
    for i in np.arange(4,a.k+1):
        plot=eigplt(i,a)
        plot[np.where(plot < 1.*10**(-7))]=0.
        xplot.append(np.arange(1,plot.size+1)/(plot.size))
        eigs.append(plot)
    #result=np.array((xplot,eigs))
    result=np.array((xplot,eigs), dtype=object)
    print('eigenvalue spectra analysis complete')
    return result

def eigplt(i, a):
    """
    Plot eigenvalues from each successive RG step, averaged over all clusters
    --------------------------------------------------------------------------------------
    Inputs:
    i: number of RG step performed. shape: scalar
    --------------------------------------------------------------------------------------
    Output: cluster 1's eigenvalues from RG step i. shape: (N/(2**i),)
    
    """
    hold=[]
    for j in range(int(a.N/2**i)):
        hold.append(a.eigsnew(i, j)[0])
    hold=np.mean(np.vstack(hold), axis=0)
    return hold

def rollx(arr):
    """
    rolls indices of each neuron's times series independently
    this is the method meshulam et al used to shuffle data
    -------------------------------------------------- 
    Inputs:
    x: matrix, we want indices of rows to be rolled independently
    -------------------------------------------------------
    output: matrix with rolling procedure applied on it
    """
    x=np.zeros(arr.shape)
    num=np.random.choice(x.shape[1], size=(x.shape[0],))
    for i in range(x.shape[0]):
        x[i,:]=np.roll(arr[i,:],num[i])
    return x	

def drawpdf(dist, binz):
    
    """
    Draw probability density function from a given data set.
    ------------------------------------------------------------------------
    Inputs:
    dist: given data set. shape: (dist.size,)
    dt: bin width for pdf calculation. shape: scalar
    -----------------------------------------------------------------------
    Output: 
    x: bin locations. shape: (res.size,)
    res: probability of data being in that bin. shape:(int((max(dist)-
         min(dist))/dt),)
    """
    
    x,y = np.histogram(dist, bins=binz, density=True)
    #x=x/np.sum(x)
    for i in range(0,len(y)-1):
        y[i]=(y[i]+y[i+1])/2
    y=y[0:len(y)-1]
    return y,x
"""
calculate variance of activity at each RG step (over all clusters)
"""
def varpltover(i, a):
    """
    Calculate variance over all coarse grained variables (clusters) at RG step i.
    -------------------------------------------------------------------------------
    Inputs: 
    i: RG step
    -------------------------------------------------------------------------------
    Output: variance over all course grained variables at RG step i
    
    """
    varplot=np.var(a.pmatarr(i))
    #print(i)
    return varplot
def varplotall(a):
    varover=[]
    for i in range(a.k+1):
        varover.append(varpltover(i, a))
    result=(2**(np.arange(a.k+1)), varover)
    print('variance scaling analysis complete')
    return result

"""
Plot log probability of complete cluster silence vs cluster size
"""
def probplotall(a):
    probdata=[]
    probx=[]
    for i in range(a.k+1):
        whpr=np.where(a.pmatarr(i)==0)
        prob=np.array(whpr[1]).size
        contain=np.log(prob/a.pmatarr(i).size)
        probdata.append(contain)
        probx.append(2**i)
    probdata=(np.array(probdata))
    probx=np.array(probx)
    result=(probx, probdata)
    print('free energy scaling analysis complete')
    return result

def activemom(a):
    collectx=[]
    collect=[]
    kurtosis=[]
    for i in 2**(np.arange(1,8)):
        pmatnew=normmom(RGmom(i,a))
        result=drawpdf(pmatnew.flatten(), 100)
        collectx.append(result[0])
        collect.append(result[1])
        if i == max(2**(np.arange(1,8))):
            top=np.mean(pmatnew.flatten()**4)
            bottom=np.mean(pmatnew.flatten()**2)**2
            kurtosis.append((top/bottom))
    print('momentum space activity distribution analysis complete')
    return collectx, collect, kurtosis
    
def normmom(ppmat):
    """
    Makes the sum of squares of momentum space RG'd activity equal to 1
    ----------------------------------------------------------------------------
    Inputs: 
    ppmat: momentum space RG'd activity array.  shape: (N/l, xmax*dt*loop)
    --------------------------------------------------------------------------
    Output: normalized momentum space RG'd activity array.  shape: (N/l, xmax*dt*loop)
    kurtosis: average over time of activity^4 divided by square of average over time of activity^2
    """
    ppmatnew=np.empty(ppmat.shape)
    for i in range(ppmat.shape[0]): #enforce that sum of squares must be 1
        test=(np.sqrt(ppmat.shape[1])*ppmat[i,:])/(np.sqrt(np.sum(ppmat[i,:]**2)))
        ppmatnew[i,:]=test
    return ppmatnew


"""
For each successive RG step, calculate average autocorrelation over all 
coarse grained variables 
"""
def calccorrreal(act, interval, mode='real'): 
    """
    Calculate average autocorrelation over all cells in given activity array
    """
    nx=act.shape[1]
    lags = []
    ys=[]
    for l in range(act.shape[0]):
        autocorrs = autocorr(act[l, :], norm=mode)
        ys.append(autocorrs[1])
        lags.append(autocorrs[0])
    y=np.vstack(np.array(ys))
    y=np.mean(y, axis=0)
    x=autocorrs[0]
    result=x[int(nx)-interval:int(nx)+interval], y[int(nx)-interval:int(nx)+interval]
    return result

def correalt(a, interval, i):
    """
    Calculate average autocorrelation over all cells in given activity array a.pmatarr(i)
    """
    act=(a.pmatarr(i)).astype('float')
    x,y=calccorrreal(act, interval, mode='real')
    return x,y

def calccorrmulti(a):
    inter=600
    result=[]
    for i in range(1, a.k):
        result.append(correalt(a,inter, i)[1])
    result=np.vstack(result)
    x=correalt(a,inter, 1)[0]
    print('dynamic scaling analysis complete')
    return x, result


class recordall:
    def __init__(self, hamx, ham, probx, \
        prob, ratex, rate, rateerr, coeffx, coeff, coefferr, shuffcoeffx, shuffcoeff, eigspecx,\
        eigspec, eigspecerr, varx, var, varerr, psilx, psil, psilerr, actmomx, actmom, actmomerr, kurtosis, autocorrx,\
                autocorr, autocorrerr, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, z, zerr, phi, eta, epsilon, percell, stim, timeconst, labeltype, label):
        self.hamx=hamx
        self.ham=ham
        self.probx=probx
        self.prob=prob
        self.ratex=ratex
        self.rate=rate
        self.rateerr=rateerr
        self.coeffx=coeffx
        self.coeff=coeff
        self.coefferr=coefferr
        self.shuffcoeffx=shuffcoeffx
        self.shuffcoeff=shuffcoeff
        self.eigspecx=eigspecx
        self.eigspec=eigspec
        self.eigspecerr=eigspecerr
        self.varx=varx
        self.var=var
        self.varerr=varerr
        self.psilx=psilx
        self.psil=psil
        self.psilerr=psilerr
        self.actmomx=actmomx
        self.actmom=actmom
        self.actmomerr=actmomerr
        self.kurtosis=kurtosis
        self.autocorrx=autocorrx
        self.autocorr=autocorr
        self.autocorrerr=autocorrerr
        self.tau=tau
        self.tauerr=tauerr
        self.mu=mu
        self.muerr=muerr
        self.alpha=alpha
        self.alphaerr=alphaerr
        self.beta=beta
        self.betaerr=betaerr
        self.z=z
        self.zerr=zerr
        self.phi=phi
        self.eta=eta
        self.epsilon=epsilon
        self.percell=percell
        self.stim=stim
        self.timeconst=timeconst
        self.labeltype=labeltype
        self.label=label

def blis_gemm(X, W):
    """
    Fast matrix multiplication using blis.py
    -------------------------------------------
    Inputs:
    X: matrix shape (a,b,c)
    W: matrix shape (c,d)
    ------------------------------------------
    Output:
    X /dot W: matrix shape  (a,b,d)
    """
    contain=[]
    for i in range(X.shape[0]):
        y=gemm(X[i,:,:], W, trans1=False, trans2=False)
        contain.append(y)
    contain=np.array(contain)
    return(contain)

def linfunc(b,a,c):
    return a*b**c

def probfunc(K, a, b):
    return (a*K**b)


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

def linear(x,a):
    return -x*a

def hamhist(env):
    pl=env.h[:,:,env.latentcell].flatten()
    plott=drawpdf(pl, 100)
    return plott

def probhist(env):
    pl=1-env.P.flatten()
    plott=drawpdf(pl, 100)
    return plott

class expsum:
    def __init__(self, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, kurtosis, z, zerr):
        self.tau=tau
        self.tauerr=tauerr
        self.mu=mu
        self.muerr=muerr
        self.alpha=alpha
        self.alphaerr=alphaerr
        self.beta=beta
        self.betaerr=betaerr
        self.kurtosis=kurtosis
        self.z=z
        self.zerr=zerr

def orderplot(allo):
    """
    Force the recordall object to have data sets in order of parameter sweep value
    Look at placerg.objects to find structure of recordall object.
    For example, suppose you do a parameter sweep of the parameter 'epsilon' and record the data
    from all your simulations into a single recordall object. The analysis notebooks take this object and 
    generate plots from each attribute of the recordall object. You want each of your subplots to appear in order 
    of increasing epsilon. So use this function as demonstrated in the analysis notebooks to do that!
    -------------------------------------------
    Inputs:
    allo: a recordall object
    """
    subs=[allo.hamx, allo.ham, allo.probx, \
        allo.prob, allo.ratex, allo.rate, allo.rateerr, allo.coeffx, allo.coeff, allo.coefferr, allo.shuffcoeffx, \
        allo.shuffcoeff, allo.eigspecx,\
        allo.eigspec, allo.eigspecerr, allo.varx, allo.var, allo.varerr, allo.psilx, allo.psil, allo.psilerr,\
        allo.actmomx, allo.actmom, allo.actmomerr, allo.kurtosis, allo.autocorrx,\
        allo.autocorr, allo.autocorrerr, allo.tau, allo.tauerr, allo.mu, allo.muerr, allo.alpha, allo.alphaerr, \
        allo.beta, allo.betaerr, allo.z, allo.zerr, allo.timeconst]
    sorts=allo.label
    argsorts=np.argsort(allo.label)
    for i in range(len(subs)):
        subs[i]=[x for _,x in sorted(zip(sorts,subs[i]))]
    hamx, ham, probx, \
        prob, ratex, rate, rateerr, coeffx, coeff, coefferr, shuffcoeffx, shuffcoeff, eigspecx,\
        eigspec, eigspecerr, varx, var, varerr, psilx, psil, psilerr, actmomx, actmom, actmomerr, kurtosis, \
        autocorrx, autocorr, autocorrerr, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, z, zerr, timeconst = subs
    allo.hamx=hamx
    allo.ham=ham
    allo.probx=probx
    allo.prob=prob
    allo.ratex=ratex
    allo.rate=rate
    allo.rateerr=rateerr
    allo.coeffx=coeffx
    allo.coeff=coeff
    allo.coefferr=coefferr
    allo.shuffcoeffx=shuffcoeffx
    allo.shuffcoeff=shuffcoeff
    allo.eigspecx=eigspecx
    allo.eigspec=eigspec
    allo.eigspecerr=eigspecerr
    allo.varx=varx
    allo.var=var
    allo.varerr=varerr
    allo.psilx=psilx
    allo.psil=psil
    allo.psilerr=psilerr
    allo.actmomx=actmomx
    allo.actmom=actmom
    allo.actmomerr=actmomerr
    allo.kurtosis=kurtosis
    allo.autocorrx=autocorrx
    allo.autocorr=autocorr
    allo.autocorrerr=autocorrerr
    allo.tau=tau
    allo.tauerr=tauerr
    allo.mu=mu
    allo.muerr=muerr
    allo.alpha=alpha
    allo.alphaerr=alphaerr
    allo.beta=beta
    allo.betaerr=betaerr
    allo.z=z
    allo.zerr=zerr
    allo.timeconst=timeconst
    allo.label=np.array(allo.label)[argsorts]
    allo.phi=np.array(allo.phi)[argsorts]
    allo.eta=np.array(allo.eta)[argsorts]
    allo.epsilon=np.array(allo.epsilon)[argsorts]
    allo.percell=np.array(allo.percell)[argsorts]
    allo.stim=np.array(allo.stim)[argsorts]

def gaussian(x,b,c):
    # make function of mean and standard dev, normalize it
    return (1/(c*np.sqrt(2*np.pi)))*np.exp((-(x-b)**2)/(2*c**2))
   