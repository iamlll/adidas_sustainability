import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit
from matplotlib.ticker import MaxNLocator, LogLocator
from tools import parse_CSV
import sys

def FindAnaPs(K,c,eps,e_s,m,f,gamma=0.4,opt='lin'):
    if opt == 'lin':
        pn = 0.5*(1/gamma+c+K)
        ps = 1/(2*gamma)+ (c+K+f*(eps+K*(e_s-1)))*0.5/(1+(m-1)*f)
    else:
        pn = 1/gamma+c+K
        ps = 1/gamma+ (c+K+f*(eps+K*(e_s-1)))/(1+(m-1)*f)
    return pn, ps

def FindAnaProfits(K,c,eps,e_s,m,f,gamma=0.4):
    pn,ps = FindAnaPs(K,c,eps,e_s,m,f,gamma,opt='lin')
    Pn = 1/(4*gamma)* (1-gamma*(c+K))**2
    Ps = 1/(4*gamma*(1+f*(m-1))) * (-1+ gamma*(c+K) + f*(1-m+gamma*(eps+K*(e_s-1))))**2
    return Pn, Ps

def FindKs(c,eps,e_s,m,f,gamma=0.4,opt='lin'):
    if opt == 'lin':
        prefactor = gamma*(2*e_s-(m+1)+f*(1-e_s)**2)
        K1 = (1+f*(m-1))*(e_s-1) -gamma*(eps*(1+f*(e_s-1)) +c*(e_s-m))
        K2 = np.sqrt((e_s*(1-c*gamma)+gamma*(c+eps)-m)**2 *(1+(m-1)*f))
        Kp = 1/prefactor*(K1+K2)
        Km = 1/prefactor*(K1-K2)
        return Km,Kp
    else: #exponential
        return ((1-m)*c+eps)/(m-e_s) +(f*(1-m)-1)/(f*gamma*(m-e_s)) *np.log(1+f*(m-1))

def PlotProfitVsK_ana(Pn,Ps,r):
    #Ps.e_s = 0
    Ks = np.linspace(.8,1.1,1000)
    Km,Kp = FindKs(Ps.c,Ps.eps,Ps.e_s,Ps.m,Ps.f_s,r.gamma,opt='lin')
    pn,ps = FindAnaPs(Ks,Ps.c,Ps.eps,Ps.e_s,Ps.m,Ps.f_s,r.gamma,opt='lin')
    pmax = 1/r.gamma
    Ln = Pn.val(pn,r,Ks)
    Ls = Ps.val(ps,r,Ks)
    Ln2 = np.where(pn <= pmax, Pn.val(ps,r,Ks),0)
    Ls2 = np.where(ps <= pmax, Ps.val(ps,r,Ks),0)
    fig, ax = plt.subplots(figsize=(7,5))
    diff = Ls-Ln
    diff2 = Ls2-Ln2
    ax.plot(Ks,diff,label='$P_S-P_N$')
    ax.plot(Ks,diff2,label='$P_S-P_N$ (2)')
    print(diff[:10])
    print(diff2[:10])
    
    ax.plot(Ks,pn,label='$p_n^*$')
    ax.plot(Ks,ps,label='$p_s^*$')
    ax.plot(Ks,[pmax]*len(Ks),label='$p_{max}=1/\gamma$') 
    #find crossover point of p_n, p_s & p_max = 1/gamma
    intersect_s = np.argwhere(np.diff(np.sign(1/r.gamma-ps))).flatten()
    intersect_n = np.argwhere(np.diff(np.sign(1/r.gamma-pn))).flatten()
    print(intersect_s)
    print(intersect_n)
    #ax.axvline(ps[intersect_s])
    #ax.axvline(pn[intersect_s])
    
    ax.set_ylabel('$P$')
    ax.set_xlabel('carbon tax K')
    ax.plot(Km,0,'bo',label='$K_{min}^-$')
    ax.plot(Kp,0,'ro',label='$K_{min}^+$')
    #ax.plot([Km,Kp],[0,0],'bo',label='$K_{min}$')
    print(Km,Kp)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    #ax.set_ylim(bottom=-.0005)
    ax.legend()
    fig.suptitle('$(c,\epsilon,e_s,m,f,\gamma) = (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)$' %(Ps.c,Ps.eps,Ps.e_s,Ps.m,Ps.f_s,r.gamma))
    plt.tight_layout()
    plt.show()

def Plot4D(size=3,e_s=0.,gamma=1.,opt='K',**kwargs):
    ''' 
    Generate data for Kmin as a function of c and epsilon, at different values of f and m (3x3 plots)
    Set gamma = 1 (i.e. nondimensionalize the rate / p wrt p_max = 1/gamma for the linear case)
    Set e_s = 0 (impossible but pretend there is such a material that incurs no environmental cost whatsoever -- next step can be to make plots with differing values of e_s)
    0 <= c <= 1 so that 0 <= p <= 1 (since e.g. p_n = 1+c when K=0)
    '''
    from fun_model6 import ProfDensityNS, ProfDensityS, LinRate
    #Check that all regions have p_n and p_s <= p_max = 1/gamma, otherwise use the alternative K soln where p_{n,s} = p_max so that P_{N,S} = 0
    cs = np.linspace(0,1,200)
    epss = np.linspace(0,1,100)
    #ms = np.linspace(0.7,2,size)
    #fs = np.linspace(0.01,1,size)
    fs = [0.01,0.1,0.5]
    ms = [1.01,1.1,1.5]
    X,Y = np.meshgrid(cs,epss)
    r = LinRate(gamma)
    '''
    fig2, ax2 = plt.subplots(figsize=(7,5))
    K2m,K2p = FindKs(0.1,epss,e_s,0.3,0,gamma)
    ax2.plot(epss,K2m,label='-')
    ax2.plot(epss,K2p,label='+')
    ax2.legend()
    ax2.set_xlabel('eps')
    ax2.set_ylabel('Kmin')
    '''
    cmin = 0
    cmax = 0
    fig, axs = plt.subplots(size,size,figsize=(8,6))
    for i,j in product(range(size),range(size)):
        print(ms[i],fs[j])
        Km,Kp = FindKs(X,Y,e_s,ms[i],fs[j],gamma,opt='lin')
        #the selected K_min value should be >=0 but should be the minimum between the two Kmin solutions
        #Kmins= np.maximum(0, np.minimum(Km,Kp)) 
        Kmins= np.maximum(0, Kp) 
        pn,ps = FindAnaPs(Kmins,X,Y,e_s,ms[i],fs[j],gamma,opt='lin')
        Pn,Ps = FindAnaPs(Kmins,X,Y,e_s,ms[i],fs[j],gamma)
        if opt == 'Pn': #plot nonsustainable profit density on Z axis
            Z = np.ma.masked_where(pn > 1/gamma, Pn)
            cbarlabel = '$P_N$'
        elif opt == 'Ps': #plot nonsustainable profit density on Z axis
            Z = np.ma.masked_where(ps > 1/gamma, Ps)
            cbarlabel = '$P_S$'
        elif opt == 'diff': #plot nonsustainable profit density on Z axis
            Z = np.where((pn <= 1/gamma) & (ps <= 1/gamma) & (Ps-Pn > 0), Ps- Pn,np.nan)
            Z = np.ma.masked_invalid(Z)
            cbarlabel = '$P_S-P_N$'
        else: #plot Kmin on Z axis
            altKs = (Y-(ms[i]-1)*1/gamma)/(1-e_s) #found from setting p_n = p_s = p_max (I believe this returns 0 profit though)
            cond = (pn <= 1/gamma) & (ps <= 1/gamma) & (Ps- Pn > 0)
            Z = np.where(cond, Kmins, altKs)
            cbarlabel = '$K_{min}$'
        print(Z.min(),Z.max())
        if 'cmax' in kwargs:
            cmax = kwargs['cmax']
        else:
            cmax = np.maximum(cmax,Z.max())
        if 'cmin' in kwargs:
            cmin = kwargs['cmin']
        else:
            cmin = np.maximum(cmin,np.amin(Z))
        cf = axs[i,j].contourf(X,Y,Z, levels = MaxNLocator(nbins=15).tick_values(cmin,cmax))
        cbar = fig.colorbar(cf, ax=axs[i,j])
        axs[i,j].set_xlabel('$c$')
        axs[i,j].set_ylabel('$\epsilon$')
        cbar.ax.set_ylabel(cbarlabel)
        axs[i,j].set_title('$(m,f) = (%.2f, %.2f)$' %(ms[i],fs[j]))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product
    from fun_model6 import ProfDensityNS, ProfDensityS, LinRate, ExpRate

    gamma = 1.
    c = 0.1
    pmax = 10.
    #rate = ExpRate(gamma)
    rate = LinRate(gamma)
    pmax = 1/gamma
    Pn = ProfDensityNS(c=c,pmax=pmax) 
    Ps = ProfDensityS(c=c,pmax=pmax,s=0.2,eps=0.3) 
    #PlotProfitVsK_ana(Pn,Ps,rate)
    #PlotK(csvname)
    #AnalyticK(gamma,c=c,opt='lin')
    Plot4D(3,e_s=0.,gamma=1.,opt='K')
    #print(FindKs(c=.1,eps=0.1,e_s=0.3,m=0.5,f=0.1,gamma=0.4,opt='lin'))
    #print(FindKs(c=.1,eps=.1,e_s=0.3,m=0.5,f=0.4,gamma=0.4,opt='lin'))
    #print(FindKs(c=.1,eps=.1,e_s=0.3,m=0.5,f=0.9,gamma=0.4,opt='lin'))
