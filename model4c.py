import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit
from matplotlib.ticker import MaxNLocator, LogLocator
from tools import parse_CSV
import sys

def PlotProfDensity(profitfxn, f_s):
    ps = np.linspace(0,profitfxn.pmax,100)
    ms = np.linspace(0,2,80)
    P, M = np.meshgrid(ps,ms,sparse=False)
    L = profitfxn.val(P,M,f_s)
    #R = fn.GetRate(P,S)
    #L = np.where(R>=0, fn.Value(P,S), 1E-5)
    fig, ax = plt.subplots(figsize=(7,5.5))
    #print(L.min(),L.max())
    cf = ax.contourf(P,M,L, levels = MaxNLocator(nbins=20).tick_values(L.min(),L.max()))
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel('$P$')
    ax.set_xlabel('unit selling price (p)')
    ax.set_ylabel('sustainability premium (m)')
    fig.suptitle('$K_s = $' + str(profitfxn.Ks) + ', $K_n = $' + str(profitfxn.Kn))
    plt.tight_layout()
    plt.show()

def Case1(P,f,r):
    '''mu_i = 0 for all i, just solve
    partial_p P = 0
    partial_m P = 0
    '''
    p,m = fsolve(lambda x: P.grad(x[0],x[1],f,r), [P.pmax/2,2]) 
    if P.cond(p,m,f): return p,m
    else: return -2,-2

def Case2(P,f,r):
    m = f.mmax
    p = fsolve(lambda x: P.grad(x,m,f,r)[0], [P.pmax/2])[0] 
    
    if P.cond(p,m,f) == False: return -2,-2
    mu1 = P.grad(p,m,f,r)[1]
    if (mu1 >=0): return p,m
    else: return -2,-2

def Case3(P,f,r):
    m = f.mmax
    p = P.pmax
    if P.cond(p,m,f) == False: return -2,-2
    mu2,mu1 = P.grad(p,m,f,r)
    if (mu1 >=0) & (mu2 >= 0): return p,m
    else: return -2,-2

def Case4(P,f,r):
    p = P.pmax
    m = fsolve(lambda x: P.grad(p,x,f,r)[1], [5])[0]    
    if P.cond(p,m,f) == False:
        return -2,-2
    mu2 = P.grad(p,m,f,r)[0]
    if mu2 >= 0: return p,m
    else: return -2,-2

def Case5(P,f,r):
    #first solve g3, g3' = 0 for p and m
    p,m = fsolve(lambda x: P.GetConstraint(x[0],x[1],f,r),[P.pmax/2,2])
    if P.cond(p,m,f) == False: return -2,-2
    #Next, solve the KKT conditions to get mu3 and mu3'
    mu3, mu3p = fsolve(lambda x: P.grad(p,m,f,r) - x[0]*P.GradConstraint(p,m,f,r)[0] - x[1]*P.GradConstraint(p,m,f,r)[1], [1,1])
    if (mu3 >= 0) & (mu3p >= 0): return p,m
    else: return -2,-2

def FindStuff(args):
    P,f, r,Kn,Ks,n = args
    P.Ks = Ks
    P.Kn = Kn
    
    if n == 1: p,m = Case1(P,f,r)
    elif n == 2: p,m = Case2(P,f,r)
    elif n == 3: p,m = Case3(P,f,r)
    elif n == 4: p,m = Case4(P,f,r)
    elif n == 5: p,m = Case5(P,f,r)
    else: p,m = -2,-2
    return n, Kn,Ks, r.gamma, P.beta,P.c0, P.pmax, p,m, r.val(p), P.val(p,m,f,r), P.E(p,m,f,r)[1], P.E(p,m,f,r)[0], P.E(p,m,f,r)[-1],P.normval(p,m,f,r)

def FindAllCases(args):
    '''Find max profit from all cases'''
    P,f,r,Kn,Ks = args

    ns = np.arange(5)+1
     
    info = np.array([FindStuff((P,f,r,Kn,Ks,n)) for n in ns],dtype=tuple)
    vals = info[:,10]
    idx = np.where(vals==vals.max())[0]
    return info[idx,:][0]

def PoolParty(P,f,r,Ksmax=5,Knmax=5,num=np.nan):
    Kns = np.linspace(0,Knmax,100)
    Kss = np.linspace(0,Ksmax,100)
    df={}
    quantities = ['case','Kn','Ks','gamma','beta','c0','pmax','p','m','r','P','E_n', 'E_s','Etot','P/E']

    for i in quantities:
        df[i]=[]

    prefix = 'model4b'

    if np.isnan(num) == False: 
        csvname = prefix + "_case" + str(num) + ".csv"
    else:
        csvname = prefix + ".csv"

    tic = time.perf_counter()
     
    with multiprocessing.Pool(processes=4) as pool:
        if np.isnan(num) == False: 
            job_args = [(P,f,r,Kn,Ks,num) for Kn,Ks in product(Kns,Kss)]
            results = pool.map(FindStuff, job_args)
        else:
            job_args = [(P,f,r,Kn,Ks) for Kn,Ks in product(Kns,Kss)]
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
    beta = df['beta'].values[0]
    c0 = df['c0'].values[0]
    colnames = ['Kn','Ks','P/E']
    colnames1b = ['Kn','Ks','P']
    colnames1c = ['Kn','Ks','Etot']
    colnames1d = ['Kn','Ks','m']
    Kns,Kss,P = parse_CSV(df,colnames)
    #Kns,Kss,P = parse_CSV(df,colnames1b)
    _,_,Ms = parse_CSV(df,colnames1d)
    colnames2 = ['Kn','Ks','case']
    Kns,Kss,Zcase = parse_CSV(df,colnames2)
    #L = np.where((P>=0) , P, np.nan)
    #maskedL = ma.masked_invalid(L)
    #L3 = np.where((P>= 0) & (E > 0), Zcase, np.nan)
    #maskedL3 = ma.masked_invalid(L3)
    cpmin = 0.
    cf = ax[0].contourf(Kns,Kss,P, levels = MaxNLocator(nbins=20).tick_values(cpmin,P.max()))
    cf3 = ax[2].contourf(Kns,Kss,Ms, levels = MaxNLocator(nbins=20).tick_values(Ms.min(),Ms.max()))

    nbins_case = np.unique(Zcase)
    print(nbins_case)
    nbins_case = int(nbins_case.max()-nbins_case.min())
    cf2 = ax[1].contourf(Kns,Kss,Zcase, levels = MaxNLocator(nbins=nbins_case+1).tick_values(Zcase.min(),Zcase.max()))
    cbar = fig.colorbar(cf, ax=ax[0])
    cbar2 = fig.colorbar(cf2, ax=ax[1])
    cbar3 = fig.colorbar(cf3, ax=ax[2])
    fig.suptitle("$\gamma= " + str(gamma) + "$, $\\beta = " + str(beta) + "$, $c_0 = " + str(c0) + "$")
    cbar.ax.set_ylabel('$' + colnames[-1] + '$')
    cbar2.ax.set_ylabel(colnames2[-1])
    cbar3.ax.set_ylabel(colnames1d[-1])

    fig.supxlabel('$K_n$')
    fig.supylabel('$K_s$')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product
    from fun_model4c import ProfDensity, WTP_cubic, LinRate, ExpRate

    fs = WTP_cubic(mmax=10)
    #fs = WTP_linear()
    gamma = 0.4
    Ks = 1.
    Kn = 1
    #rate = LinRate(gamma=gamma)
    #P = ProfDensity(pmax=1/gamma,K_s=Ks, K_0=Kn) 
    rate = ExpRate(gamma=gamma)
    P = ProfDensity(pmax=10,K_s=Ks, K_0=Kn) 

    #betagammaPD(p_max,B,generate=True)
    #PlotProfDensity(P,fs)
    #print(Case3(P,fs))

    csvname = PoolParty(P,fs,rate) 
    #csvname = sys.argv[1]
    print(csvname)
    KnKsPD(csvname)
