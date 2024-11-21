#M. Shane Li : 11.16.2024
#This file is the for loop file: all functions that require very long and complicated for loops I put in here. There's s loopall, which goes through several simulations, and bootloop and bootfunc, which are called by loopall. 
#runfunc.py
import numpy as np
import glob
from simulate import *
from analyze import *
from objects import *
from scipy.optimize import curve_fit


#im done working tonight: 11.16.2024
#here's what i've done and am doing
#through first through sixth, i've started isolating all the functions that we need from morrell, and afterwards I will reorganize into something more intuitive
#right now i am working through dependencies. I made it to the bootloop function called in loopall, and added that afterwards. Then I added bootfunc, but im really not sure what these boot functions are needed. i will investigate etomorrow i suppose. 

#update 11.17.2024
#i think i've finished adding everything. here is how i understand boot: loopall goes through a set of simulations and collects the analysis.
#for every individual simulation, loopall calls bootloop, which runs infoset on the simulation, bootfunc to do the analysis on those simulations (since infoset just does the clustering) and then saves the results, including the multiple autocorrelation and eigenspectrums per clustering. bootfunc, because of the analysis, requires a bunch of different functions that are below. at the end, recordall is just a way to save analysis details
#
def loopall(arra, arrenv, keyword):

    hamx=[]
    ham=[]

    probx=[]
    prob=[]

    ratex=[]
    rate=[]
    rateerr=[]

    coeffx=[]
    coeff=[]
    coefferr=[]

    shuffcoeffx=[]
    shuffcoeff=[]

    eigspecx=[]
    eigspec=[]
    eigspecerr=[]

    varx=[]
    var=[]
    varerr=[]

    psilx=[]
    psil=[]
    psilerr=[]

    actmomx=[]
    actmom=[]
    actmomerr=[]

    autocorrx=[]
    autocorr=[]
    autocorrerr=[]

    tau=[]
    tauerr=[]

    mu=[]
    muerr=[]

    alpha=[]
    alphaerr=[]

    beta=[]
    betaerr=[]

    kurtosis = []

    z=[]
    zerr=[]

    epsilon=[]
    eta=[]
    phi=[]
    percell=[]


    stim=[]
    timeconst=[]

    labeltype=[]
    label=[]



    for i in range(len(arra)):
        #go through every looped version of a simulation
        #bootloop, which 
        boot=bootloop(arra[i], arrenv[i])
        # define object names we want to load in
        aname= arra[i]
        envname=arrenv[i]

        # load in objects
        env=load_object(envname)
        a=load_object(aname)

        # histogram hamiltonians
        plott=hamhist(env)
        hamx.append(plott[0])
        ham.append(plott[1])

        # histogram probability dist
        plott=probhist(env)
        probx.append(plott[0])
        prob.append(plott[1])

        # histogram cell rate rank
        plott=cellraterank(env)
        ratex.append(plott[0])
        rate.append(plott[1])
        rateerr.append(boot.rate)

        # histogram pairwise correlations
        plott=corrcoefhist(env)
        coeffx.append(plott[0][0])
        coeff.append(plott[0][1])
        coefferr.append(boot.coeff)

        # histogram pairwise correlations, shuffled
        shuffcoeffx.append(plott[1][0])
        shuffcoeff.append(plott[1][1])
  
        # eigenvalue spectra
        xplot,plot=eigplotall(a)
        eigspecx.append(xplot)
        eigspec.append(plot)
        eigspecerr.append(boot.eigspec)
        fitx=[]
        fity=[]
        for m in range(len(xplot)):
            fitx.append(xplot[m][:int(xplot[m].size)])
            fity.append(plot[m][:int(plot[m].size)])
        fitx=np.hstack(fitx)
        fity=np.hstack(fity)
        popt, pcov = curve_fit(linfunc, fitx, np.real(fity))
        mu.append(popt)
        muerr.append(boot.mu)

        # variance over coarse grained variables
        plott=varplotall(a)
        varx.append(plott[0])
        var.append(plott[1])
        varerr.append(boot.var)
        popt, pcov = curve_fit(linfunc, plott[0][:4], plott[1][:4])
        alpha.append(popt)
        alphaerr.append(boot.alpha)

        # log p(silence)
        plott=probplotall(a)
        psilx.append(plott[0])
        psil.append(plott[1])
        psilerr.append(boot.psil)
        wh=np.where(np.isfinite(plott[1]) == True)
        x=plott[0][wh]
        y=plott[1][wh]
        popt, pcov = curve_fit(probfunc,x, y)
        beta.append(popt)
        betaerr.append(boot.beta)

        # activity, momentum space
        x, plott, k=activemom(a)
        actmomx.append(x)
        actmom.append(plott)
        actmomerr.append(boot.actmom)
        kurtosis.append(k)
    
        x,result= calccorrmulti(a)
        autocorrx.append(x)
        autocorr.append(result)
        autocorrerr.append(boot.autocorr)

        xnew=(x[int(result.shape[1]/2)-1:int(result.shape[1]/2)+1])
        taus=[]
        for l in range(result.shape[0]):
            y=np.log(result[l, int(result.shape[1]/2)-1:int(result.shape[1]/2)+1])
            y[np.where(np.isfinite(y)==False)]=0.
            popt, pcov = curve_fit(linear, xnew, y, maxfev=20000)
            taus.append(popt[0])

        taus=1./np.array(taus).flatten()
        tau.append(taus)
        tauerr.append(boot.tau)

        popt, pcov = curve_fit(linfunc, 2**np.arange(1,a.k)[:4],\
                           taus[:4])
        z.append(popt)
        zerr.append(boot.z)

        # record parameters
        eta.append(env.eta)
        phi.append(env.phi)
        epsilon.append(env.epsilon)
        percell.append(env.percell)
        timeconst.append(env.timeconst)
        stim.append(env.nstim)

        if keyword == 'time':
            # labels for plots
            labeltype.append('time constant')
            label.append(env.timeconst[0])
        
        if keyword == 'type':
            # labels for plots
            labeltype.append('cell type')
            label=['both', 'place', 'none', 'no latent']

        if keyword == 'stim':
            # labels for plots
            labeltype.append('# of stimuli')
            label.append(env.nstim)

        if keyword == 'eta':
            # labels for plots
            labeltype.append('eta')
            label.append(env.eta)

        if keyword == 'phi':
            # labels for plots
            labeltype.append('phi')
            label.append(env.phi)

        if keyword == 'epsilon':
            # labels for plots
            labeltype.append('epsilon')
            label.append(env.epsilon)

        if keyword == 'percell':
            # labels for plots
            labeltype.append('p')
            label.append(env.percell)

        print('loop ' + str(i+1)+'/'+str(len(arra))+ ' complete, yipeee!')


    pltall=recordall(hamx, ham, probx, \
        prob, ratex, rate, rateerr, coeffx, coeff, coefferr, shuffcoeffx, shuffcoeff, eigspecx, eigspec, eigspecerr, varx, var, varerr, psilx, psil, psilerr, actmomx, actmom, actmomerr, kurtosis, autocorrx, autocorr, autocorrerr, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, z, zerr, phi, eta, epsilon, percell, stim, timeconst, labeltype, label)
    exp = expsum(tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, kurtosis, z, zerr)
    return pltall, exp

def bootloop(aname, envname):

    rate=[]

    coeff=[]

    eigspec0=[]
    eigspec1=[]
    eigspec2=[]
    eigspec3=[]
    eigspec4=[]

    var=[]

    psil=[]

    actmom=[]

    autocorr0=[]
    autocorr1=[]
    autocorr2=[]
    autocorr3=[]
    autocorr4=[]
    autocorr5=[]
    autocorr6=[]
    tau=[]

    mu=[]
    alpha=[]
    beta=[]
    z=[]

    env=load_object(envname)
    a=load_object(aname)
    for i in range(len(env.boots)):
        env.pmat=env.boots[i]
        a=infoset(env.N, env.pmat, a.k)
        boot=bootfunc(a, env)

        rate.append(boot.rate)

        coeff.append(boot.coeff)

        eigspec0.append(boot.eigspec[0][0])
        eigspec1.append(boot.eigspec[0][1])
        eigspec2.append(boot.eigspec[0][2])
        eigspec3.append(boot.eigspec[0][3])
        #eigspec4.append(boot.eigspec[0][4])

        var.append(boot.var)

        psil.append(boot.psil)

        actmom.append(boot.actmom)


        autocorr0.append(boot.autocorr[0][0])
        autocorr1.append(boot.autocorr[0][1])
        autocorr2.append(boot.autocorr[0][2])
        autocorr3.append(boot.autocorr[0][3])
        autocorr4.append(boot.autocorr[0][4])
        autocorr5.append(boot.autocorr[0][5])
        #autocorr6.append(boot.autocorr[0][6])
        tau.append(boot.tau)

        mu.append(boot.mu)
        alpha.append(boot.alpha)
        beta.append(boot.beta)
        z.append(boot.z)

    rate=np.std(np.vstack(rate), axis=0)
    coeff=np.std(np.vstack(coeff), axis=0)

    eigspec0=np.std(np.vstack(eigspec0), axis=0)
    eigspec1=np.std(np.vstack(eigspec1), axis=0)
    eigspec2=np.std(np.vstack(eigspec2), axis=0)
    eigspec3=np.std(np.vstack(eigspec3), axis=0)
    #eigspec4=np.std(np.vstack(eigspec4), axis=0)

    var=np.std(np.vstack(var), axis=0)
    psil=np.std(np.vstack(psil), axis=0)
    actmom=np.std(np.vstack(actmom), axis=0)

    autocorr0=np.std(np.vstack(autocorr0), axis=0)
    autocorr1=np.std(np.vstack(autocorr1), axis=0)
    autocorr2=np.std(np.vstack(autocorr2), axis=0)
    autocorr3=np.std(np.vstack(autocorr3), axis=0)
    autocorr4=np.std(np.vstack(autocorr4), axis=0)
    autocorr5=np.std(np.vstack(autocorr5), axis=0)
    #autocorr6=np.std(np.vstack(autocorr6), axis=0)
    tau=np.std(np.vstack(tau), axis=0)

    mu=np.std(np.vstack(mu), axis=0)
    alpha=np.std(np.vstack(alpha), axis=0)
    beta=np.std(np.vstack(beta), axis=0)
    z=np.std(np.vstack(z), axis=0)
    #eigspec=[eigspec0, eigspec1, eigspec2, eigspec3, eigspec4]
    eigspec=[eigspec0, eigspec1, eigspec2, eigspec3]
    autocorr=[autocorr0, autocorr1, autocorr2, autocorr3, autocorr4, autocorr5]
    #autocorr=[autocorr0, autocorr1, autocorr2, autocorr3, autocorr4, autocorr5, autocorr6]
    pltall=bootstrap(rate, coeff, \
        eigspec,var, psil, actmom, \
                autocorr, tau, mu, alpha, beta, z)
    print('bootstrap competed')
    return pltall  

def bootfunc(a, env):

    rate=[]

    coeff=[]

    eigspec=[]

    var=[]

    psil=[]

    actmom=[]

    autocorr=[]
    
    tau=[]

    
    mu=[]
    alpha=[]
    beta=[]
    z=[]


    # histogram cell rate rank
    plott=cellraterank(env)
    rate.append(plott[1])

    # histogram pairwise correlations
    plott=corrcoefhist(env)
    coeff.append(plott[0][1])

    xplot,plot=eigplotall(a)
    eigspec.append(plot)
    fitx=[]
    fity=[]
    for m in range(len(xplot)):
        fitx.append(xplot[m][:int(xplot[m].size/2)])
        fity.append(plot[m][:int(plot[m].size/2)])
    fitx=np.hstack(fitx)
    fity=np.hstack(fity)
    popt, pcov = curve_fit(linfunc, fitx, np.real(fity))
    mu.append(popt[1])

    plott=varplotall(a)
    var.append(plott[1])    
    popt, pcov = curve_fit(linfunc, plott[0][:4], plott[1][:4])
    alpha.append(popt[1])

    plott=probplotall(a)
    wh=np.where(np.isfinite(plott[1]) == True)
    x=plott[0][wh]
    y=plott[1][wh]
    psil.append(y)
    popt, pcov = curve_fit(probfunc,x, y)
    beta.append(popt[1])

    x, plott,k=activemom(a)
    actmom.append(plott)

    x, result= calccorrmulti(a)
    autocorr.append(result)

    xnew=(x[int(result.shape[1]/2)-1:int(result.shape[1]/2)+1])
    taus=[]
    for l in range(result.shape[0]):
        y=np.log(result[l, int(result.shape[1]/2)-1:int(result.shape[1]/2)+1])
        y[np.where(np.isfinite(y)==False)]=0.
        popt, pcov = curve_fit(linear, xnew, y, maxfev=20000)
        taus.append(popt[0])
    taus=1/np.array(taus).flatten()
    tau.append(taus)
    popt, pcov = curve_fit(linfunc, 2**np.arange(2,a.k)[:3],\
                           taus[:3])
    z.append(popt[1])

    pltall=bootstrap(rate, coeff, \
        eigspec,var, psil, actmom, \
                autocorr, tau, mu, alpha, beta, z)
    return pltall


def globfunc(arra, arrenv, name_all, name_sum, labelname):
    arra=sorted(glob.glob(arra))
    print(str(len(arra)) +' analysis objects found')
    arrenv=sorted(glob.glob(arrenv))
    print(str(len(arrenv)) +' simulation objects found')
    pltall, expall= loopall(arra, arrenv, labelname)
    save_object(pltall, name_all)
    save_object(expall, name_sum)