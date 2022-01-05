import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit
from matplotlib.ticker import MaxNLocator, LogLocator
from tools import parse_CSV
import sys

def PlotProfDensity(profitfxn, r):
    ps = np.linspace(0,profitfxn.pmax,100)
    L = profitfxn.val(ps,r)
    fig, ax = plt.subplots(figsize=(7,5.5))
    ax.plot(ps,L)
    idx = np.where(L == L.max())[0]
    print("p_c: " + str(ps[idx]) + "\tPmax: " + str(L.max()))
    ax.set_ylabel('$P$')
    ax.set_xlabel('unit selling price (p)')
    fig.suptitle('$K_s = $' + str(profitfxn.Ks) + ', $K_n = $' + str(profitfxn.Kn))
    plt.tight_layout()
    plt.show()

def Case1(P,r):
    '''mu_i = 0 for all i, just solve
    partial_p P = 0
    '''
    p = fsolve(lambda x: P.grad(x,r), [P.pmax/2])[0] 
    if P.cond(p,r): return p
    else: return -2

def Case3(P,r):
    p = P.pmax
    if P.cond(p,r) == False: return -2
    mu3 = P.grad(p,r)
    if (mu3 >=0): return p
    else: return -2

def FindStuff(args):
    P, r,Kn,Ks,n = args
    P.Ks = Ks
    P.Kn = Kn
    
    if n == 1: p = Case1(P,r)
    elif n == 3: p = Case3(P,r)
    else: p = -2
    return n, Kn,Ks, r.gamma, P.eps, P.m,P.f_s,P.pmax, p, r.val(p), P.E(p,r), P.val(p,r)

def FindAllCases(args):
    '''Find max profit from all cases'''
    P,r,Kn,Ks = args

    ns = [1,3]
     
    info = np.array([FindStuff((P,r,Kn,Ks,n)) for n in ns],dtype=tuple)
    vals = info[:,-1]
    idx = np.where(vals==vals.max())[0]
    return info[idx,:][0]

def PoolParty(P,r,Ksmax=5,Knmax=5,num=np.nan):
    Kns = np.linspace(0,Knmax,100)
    Kss = np.linspace(0,Ksmax,100)
    df={}
    quantities = ['case','Kn','Ks','gamma','eps','m','f_s','pmax','p','r','E', 'P']

    for i in quantities:
        df[i]=[]

    prefix = 'model5'

    if np.isnan(num) == False: 
        csvname = prefix + "_case" + str(num) + ".csv"
    else:
        csvname = prefix + ".csv"

    tic = time.perf_counter()
     
    with multiprocessing.Pool(processes=4) as pool:
        if np.isnan(num) == False: 
            job_args = [(P,r,Kn,Ks,num) for Kn,Ks in product(Kns,Kss)]
            results = pool.map(FindStuff, job_args)
        else:
            job_args = [(P,r,Kn,Ks) for Kn,Ks in product(Kns,Kss)]
            results = pool.map(FindAllCases, job_args)
        for res in results:
            for name, val in zip(quantities, res):
                df[name].append(val)

    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    data.to_csv(csvname, index=False)
    print(data)
    return csvname

def KnKsPD(csvname):
    import numpy.ma as ma
    fig, ax = plt.subplots(1,3,figsize=(12,5),sharey='row',sharex='row')
    df = pd.read_csv(csvname)
    
    gamma = df['gamma'].values[0]
    eps = df['eps'].values[0]
    m = df['m'].values[0]
    fs = df['f_s'].values[0]
    colnames = ['Kn','Ks','P']
    colnames2 = ['Kn','Ks','E']
    colnames3 = ['Kn','Ks','case']
    Kns,Kss,P = parse_CSV(df,colnames)
    _,_,E = parse_CSV(df,colnames2)
    _,_,Zcase = parse_CSV(df,colnames3)
    #L = np.where((P>=0) , P, np.nan)
    #maskedL = ma.masked_invalid(L)
    #L3 = np.where((P>= 0) & (E > 0), Zcase, np.nan)
    #maskedL3 = ma.masked_invalid(L3)
    cpmin = 0.
    cf = ax[0].contourf(Kns,Kss,P, levels = MaxNLocator(nbins=20).tick_values(cpmin,P.max()))
    cf2 = ax[1].contourf(Kns,Kss,E, levels = MaxNLocator(nbins=20).tick_values(E.min(),E.max()))

    nbins_case = np.unique(Zcase)
    print(nbins_case)
    nbins_case = int(nbins_case.max()-nbins_case.min())
    cf3 = ax[2].contourf(Kns,Kss,Zcase, levels = MaxNLocator(nbins=nbins_case+1).tick_values(Zcase.min(),Zcase.max()))
    cbar = fig.colorbar(cf, ax=ax[0])
    cbar2 = fig.colorbar(cf2, ax=ax[1])
    cbar3 = fig.colorbar(cf3, ax=ax[2])
    fig.suptitle("$\gamma= $" + str(gamma) + ", $\epsilon = $" + str(eps) + ", $m= $" + str(m) + ", $f_s =" + str(fs) + "$")
    cbar.ax.set_ylabel('$' + colnames[-1] + '$')
    cbar2.ax.set_ylabel(colnames2[-1])
    cbar3.ax.set_ylabel(colnames3[-1])

    fig.supxlabel('$K_n$')
    fig.supylabel('$K_s$')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product
    from fun_model5 import ProfDensity, LinRate, ExpRate

    gamma = 0.4
    Ks = 1.
    Kn = 1
    #rate = LinRate(gamma=gamma)
    #P = ProfDensity(pmax=1/gamma,K_s=Ks, K_0=Kn) 
    rate = ExpRate(gamma=gamma)
    P = ProfDensity(pmax=10,K_s=Ks, K_0=Kn) 

    #PlotProfDensity(P,rate)

    csvname = PoolParty(P,rate) 
    KnKsPD(csvname)
