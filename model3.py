import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit
from matplotlib.ticker import MaxNLocator, LogLocator
from tools import parse_CSV
from fun_model3 import ProfDensity, ProfEnvDensity
import sys

def PlotProfDensity(fn):
    ps = np.linspace(0,fn.pmax,100)
    ss = np.linspace(0,1,80)
    P, S = np.meshgrid(ps,ss,sparse=False)
    R = fn.GetRate(P,S)
    #Rnew = np.where(R>=0, R, -1E-5)
    L = np.where(R>=0, fn.Value(P,S), 1E-5)
    fig, ax = plt.subplots(figsize=(7,5.5))
    #cf = ax.contourf(P,S,Rnew, levels = MaxNLocator(nbins=20).tick_values(0.,Rnew.max()))
    #cbar.ax.set_ylabel('$r$')
    print(L.min(),L.max())
    cf = ax.contourf(P,S,L, levels = MaxNLocator(nbins=20).tick_values(L.min(),L.max()))
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel('$f = r(p-c)$')
    ax.set_xlabel('unit price (p)')
    ax.set_ylabel('sustainability fraction (s)')
    plt.tight_layout()
    plt.show()

def Case1(f):
    '''mu_i = 0 for all i, just solve
    partial_p f = 0
    partial_s f = 0
    '''
    p0,s0 = fsolve(lambda x: f.Gradient(x[0],x[1]), [4.,0.8]) 
    if f.cond(p0,s0): return p0,s0
    else: return -1,-1

def Case2(f):
    s = 0.
    p = 0.5*(1./f.gamma + f.beta)
    if f.cond(p,s) == False: return -1,-1
    mu1 = -f.Gradient(p,s)[1]
    if mu1 >= 0: return p,s
    else: return -1,-1

def Case3(f):
    p0,s0 = f.pmax,0.
    if f.cond(p0,s0) == False: return -1,-1
    mu5,grads = f.Gradient(p0,s0)
    mu1 = -grads
    if (mu1>=0) & (mu5 >=0): return p0,s0
    else: return -1,-1

def Case4(f):
    s = 0.
    if f.beta*f.alpha >= 0.5: #fix and generalize to work for both f and f-E
        p = fsolve(lambda x: f.GetConstraint(x,s),[4.])[0]
        if f.cond(p,s) == False: return -1,-1
        gp,gs = f.GradConstraint(p,s)
        fp,fs = f.Gradient(p,s)
        mu7 = fp/gp
        mu1 = mu7*gs - fs
        if (mu1 >=0) & (mu7 >= 0): return p,s
        else: return -1,-1
    else: return -1,-1

def Case5(f):
    s=1.
    p = fsolve(lambda x: f.Gradient(x,s)[0],[4.])[0]
    if f.cond(p,s) == False: return -1,-1
    mu2 = f.Gradient(p,s)[1]
    if mu2 >= 0: return p,s
    else: return -1,-1

def Case6(f):
    p,s = f.pmax, 1.
    mu5,mu2 = f.Gradient(p,s)
    if (mu2 >=0) & (mu5 >= 0) & f.cond(p,s): return p,s
    else: return -1,-1

def Case7(f):
    s = 1.
    p = fsolve(lambda x: f.GetConstraint(x,s),4.)[0]
    if f.cond(p,s) == False: return -1,-1
    gp,gs = f.GradConstraint(p,s)
    fp,fs = f.Gradient(p,s)
    mu7 = fp/gp
    mu2 = fs - mu7*gs
    if (mu2 >=0) & (mu7 >= 0): return p,s
    else: return -1,-1
    
def Case8(f):
    p0 = f.pmax
    fs = lambda x: f.Gradient(p0,x)[1]
    s0 = fsolve(fs,[0.9])[0]
    if f.cond(p0,s0) == False: return -1,-1
    mu5 = f.Gradient(p0,s0)[0]
    if (mu5 >= 0): return p0,s0
    else: return -1,-1

def Case9(f):
    p = f.pmax
    s = fsolve(lambda x: f.GetConstraint(p,x),[0.8])[0]
    if f.cond(p,s) == False: return -1,-1
    gp,gs = f.GradConstraint(p,s)
    fp,fs = f.Gradient(p,s)
    mu7 = fs/gs
    mu5 = mu7*gp - fp
    if (mu5 >=0) & (mu7 >= 0): return p,s
    else: return -1,-1

def Case10(f):
    gp = lambda p,s: f.GradConstraint(p,s)[0]
    gs = lambda p,s: f.GradConstraint(p,s)[1]
    fp = lambda p,s: f.Gradient(p,s)[0]
    fs = lambda p,s: f.Gradient(p,s)[1]
    fun = lambda p,s: fp(p,s)/gp(p,s) - fs(p,s)/gs(p,s)
    p0,s0 = fsolve(lambda x:[fun(x[0],x[1]), f.GetConstraint(x[0],x[1])], [4.,0.9])
    if f.cond(p0,s0) == False: return -1,-1
    mu7 = fp(p0,s0)/gp(p0,s0)    
    if mu7 >= 0: return p0,s0
    else: return -1,-1

def FindStuff(args):
    n, fn, beta, gamma = args
    fn.beta = beta
    fn.gamma = gamma
    if n == 1: p,s = Case1(fn)
    elif n == 2: p,s = Case2(fn)
    elif n == 3: p,s = Case3(fn)
    elif n == 4: p,s = Case4(fn)
    elif n == 5: p,s = Case5(fn)
    elif n == 6: p,s = Case6(fn)
    elif n == 7: p,s = Case7(fn)
    elif n == 8: p,s = Case8(fn)
    elif n == 9: p,s = Case9(fn)
    elif n == 10: p,s = Case10(fn)
    else: p,s = -1,-1
    return n, fn.alpha, fn.beta, fn.gamma, fn.B,fn.pmax, p,s, fn.GetRate(p,s), fn.GetCost(s), fn.Value(p,s), fn.GetEnvCost(p,s), fn.Value(p,s)/fn.GetEnvCost(p,s)

def FindBetaGammaStuff(args):
    '''Find max profit from all cases'''
    fn,beta, gamma = args

    ns = np.arange(10)+1
     
    info = np.array([FindStuff((n, fn,beta, gamma)) for n in ns],dtype=tuple)
    fs = info[:,-3]
    idx = np.where(fs==fs.max())[0]
    return info[idx,:][0]

def PoolParty(fn,bmax=10,gmax=10,num=np.nan):
    gs = np.linspace(1E-5,gmax,100)
    bs = np.linspace(0,bmax,100)
    df={}
    quantities = ['case','alpha','beta','gamma','B','pmax','p','s','r','c','P','E', 'P/E']

    for i in quantities:
        df[i]=[]

    if np.isnan(fn.B):
        prefix = 'model3a' #NO env cost
    else: prefix = 'model3b' #WITH env cost

    if np.isnan(num) == False: 
        csvname = prefix + "_case" + str(num) + ".csv"
    else:
        csvname = prefix + ".csv"

    tic = time.perf_counter()
     
    with multiprocessing.Pool(processes=4) as pool:
        if np.isnan(num) == False: 
            job_args = [(num,fn,beta,gamma) for beta,gamma in product(bs,gs)]
            results = pool.map(FindStuff, job_args)
        else:
            job_args = [(fn,beta,gamma) for beta,gamma in product(bs,gs)]
            results = pool.map(FindBetaGammaStuff, job_args)
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

def betagammaPD(csvname):
    import numpy.ma as ma
    # 0<B<1
    fig, ax = plt.subplots(1,3,figsize=(10,5),sharey='row',sharex='row')
    df = pd.read_csv(csvname)
    
    B = df['B'].values[0]
    alpha = df['alpha'].values[0]
    colnames = ['beta','gamma','P/E']
    colnames1b = ['beta','gamma','P']
    colnames1c = ['beta','gamma','E']
    bs,gs,Z = parse_CSV(df,colnames)
    _,_,P = parse_CSV(df,colnames1b)
    _,_,E = parse_CSV(df,colnames1c)
    colnames2 = ['beta','gamma','case']
    bs,gs,Zcase = parse_CSV(df,colnames2)
    L = np.where((P>=0) & (E > 0), Z, np.nan)
    maskedL = ma.masked_invalid(L)
    L2 = np.where((P> 0) & (E > 0), np.log10(Z), np.nan)
    maskedL2 = ma.masked_invalid(L2)
    L3 = np.where((P>= 0) & (E > 0), Zcase, np.nan)
    maskedL3 = ma.masked_invalid(L3)
    cpmin = 0.
    cf = ax[0].contourf(bs,gs,maskedL, levels = MaxNLocator(nbins=20).tick_values(cpmin,maskedL.max()))
    idx = np.where(maskedL == maskedL.max())
    print(bs[idx],gs[idx])
    print(maskedL.min(),maskedL.max())
    cf3 = ax[1].contourf(bs,gs,maskedL2, levels = MaxNLocator(nbins=20).tick_values(maskedL2.min(),maskedL2.max()))

    nbins_case = np.unique(maskedL3)
    print(nbins_case)
    nbins_case = int(nbins_case.max()-nbins_case.min())
    cf2 = ax[2].contourf(bs,gs,maskedL3, levels = MaxNLocator(nbins=nbins_case+1).tick_values(maskedL3.min(),maskedL3.max()))
    cbar = fig.colorbar(cf, ax=ax[0])
    cbar2 = fig.colorbar(cf2, ax=ax[2])
    cbar3 = fig.colorbar(cf3, ax=ax[1])
    if np.isnan(B):
        colnames[-1] = 'P/C'
        fig.suptitle("$\\alpha= " + str(alpha) + "$")
    else:
        fig.suptitle("$\\alpha= " + str(alpha) + "$, $B = " + str(B) + "$")
    cbar.ax.set_ylabel('$' + colnames[-1] + '$')
    cbar2.ax.set_ylabel(colnames2[-1])
    cbar3.ax.set_ylabel('$\log_{10}(' + colnames[-1] + ')$')

    fig.supxlabel('$\\beta$')
    fig.supylabel('$\gamma$')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product

    #1/gamma >= beta
    p_max = 10
    B = 0.158
    alpha1 = 1.
    alpha2 = .1
    gmax = .1
    bmax = p_max
    #PlotKKT(csvname, p_max,case)
    
    #PlotCase10(p_max)
    f1 = ProfDensity(gamma=1.,beta=0.8,alpha=alpha1,pmax=p_max)
    f2 = ProfEnvDensity(gamma=1.,beta=0.8,alpha=alpha2,B=B,pmax=p_max)
    #betagammaPD(p_max,B,generate=True)
    #PlotProfDensity(f2)
    #print(Case10(f2))
    #print(FindBetaGammaStuff((f2,0.8,0.1)))

    csvname = PoolParty(f2,bmax,gmax) 
    #csvname = sys.argv[1]
    print(csvname)
    betagammaPD(csvname)
