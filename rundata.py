import numpy as np
from simulate import *
from analyze import *
from objects import *
from loop import *
from funcsdata import *
from scipy import sparse

#Fill out below
#choose data base and set area name
npu = np.load('/Users/mli853/Documents/NWB/data/MOspikes.npy') 
area = "MO"
#set number of course grainings
K = 7

#loads in the object
#takes number of neurons,activity array holding spike trains for all cells, and number of RG steps 
N = len(npu)
pmat = npu
a = dataset(N, pmat, K) 

#histogram cell rate rank
plott=cellraterank(a)
rate = plott[1]
ratex = plott[0]
#raterrr = 

#histogram pairwise correlations
plott=corrcoefhist(a)
coeff = plott[0][1]
coeffx = plott[0][0]
#coefferr = 

#eigenspectrum analysis
xplot,plot=eigplotalldata(a)
eigspec = plot
eigspecx = xplot
#eigspecerr = 

fitx=[]
fity=[]
for m in range(len(xplot)):
    fitx.append(xplot[m][:int(xplot[m].size/2)])
    fity.append(plot[m][:int(plot[m].size/2)])
fitx=np.hstack(fitx)
fity=np.hstack(fity)
popt, pcov = curve_fit(linfunc, fitx, np.real(fity))
mu = popt[0]
muerr = popt[1]

#variance analysis
plott=varplotalldata(a)
var = plott[1]
varx = plott[0]
#varerr = plott[2]

popt, pcov = curve_fit(linfunc, plott[0][:4], plott[1][:4])
alpha = popt[0]
alphaerr = popt[1]

#psil analysis
plott=probplotall(a)
wh=np.where(np.isfinite(plott[1]) == True)
psilx=plott[0][wh]
psil=plott[1][wh]
#psilerr 

popt, pcov = curve_fit(probfunc,psilx, psil)
beta = popt[0]
betaerr =  popt[1]

#momentum space analysis
x, plott, k = activemom(a)
actmom = plott
actmomx = x
kurtosis = k
#actmomerr
'''
#autocorrelation analysis
x, result = calccorrmulti(a)
autocorr = result
xnew=(x[int(result.shape[1]/2)-1:int(result.shape[1]/2)+1])
taus=[]
for l in range(result.shape[0]):
    y=np.log(result[l, int(result.shape[1]/2)-1:int(result.shape[1]/2)+1])
    y[np.where(np.isfinite(y)==False)]=0.
    popt, pcov = curve_fit(linear, xnew, y, maxfev=20000)
    taus.append(popt[0])
taus=1/np.array(taus).flatten()
tau = taus
popt, pcov = curve_fit(linfunc, 2**np.arange(2,8)[:3],\
                           taus[:3])
z = popt[1]
z, tau,, autocorr
'''
#put it all together
pltall=databoot(rate, ratex, coeff, coeffx,\
    eigspec, eigspecx, var, varx, psil, psilx, \
            mu, muerr,alpha, alphaerr,beta, betaerr, N, area, actmom, actmomx, kurtosis)
name_a = '/users/mli853/Documents/placerg-main/datarep/NPUltraCrossBrain/data_area{}neurons{}.pkl'.format(area, N)
save_object(pltall, name_a)