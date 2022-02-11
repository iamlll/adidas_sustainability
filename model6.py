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

def FindK2(pn,ps,rn,rs,c,eps,e_s,m,f):
    '''min K given by K > [r(pn)/r(ps) (pn-c) -(ps-c) +f(ps(1-m)+c)] / [f(1-es) -1 + r(pn)/r(ps)]
    Basically, minimize profits just wrt ps, pn for a given K, then find K_min from the above expression and compare whether K>K_min. If yes, great, if not, invalid solution.
    '''
    numer = rn/rs *(pn-c)-(ps-c) + f*(ps*(1-m)+eps)
    denom = f*(1-e_s) - 1 + rn/rs
    return numer/denom

def PlotProfitVsK_ana(Pn,Ps,r):
    Ks = np.linspace(0,1.5,200)
    Km,Kp = FindKs(Ps.c,Ps.eps,Ps.e_s,Ps.m,Ps.f_s,r.gamma,opt='lin')
    pn,ps = FindAnaPs(Ks,Ps.c,Ps.eps,Ps.e_s,Ps.m,Ps.f_s,r.gamma,opt='lin')
    Ln = Pn.val(pn,r,Ks)
    Ls = Ps.val(ps,r,Ks)
    fig, ax = plt.subplots(figsize=(7,5))
    #ax.plot(Ks,Ln,label='NS')
    #ax.plot(Ks,Ls,label='S')
    ax.plot(Ks,Ls-Ln,label='$P_S-P_N$')
    ax.set_ylabel('$P$')
    ax.set_xlabel('carbon tax K')
    ax.plot([Km,Kp],[0,0],'bo',label='$K_{min}$')
    print(Km,Kp)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_ylim(bottom=-.0005)
    ax.legend()
    fig.suptitle('$(c,\epsilon,e_s,m,f,\gamma) = (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)$' %(Ps.c,Ps.eps,Ps.e_s,Ps.m,Ps.f_s,r.gamma))
    plt.tight_layout()
    plt.show()

def AnalyticK(gamma=0.4,e_s=1-0.158*0.2,m=1.25,c=1,f=0.34,opt='lin'):
    eps = np.linspace(0,1,100)
    fig, ax = plt.subplots(2,3,figsize=(7,5.5))
    ax[0,0].plot(eps,FindK(c,eps,e_s,m,f,gamma,opt=opt))
    ax[0,0].set_title('$e_s = 1-0.158*0.2$')
    e_s = np.linspace(0,1,100) 
    eps = 0.2
    ax[0,1].plot(e_s,FindK(c,eps,e_s,m,f,gamma,opt))
    ax[0,1].set_title('$\epsilon =$' + str(eps))
    e_s = 1-0.158*0.2
    c = np.linspace(0,5,500)
    ax[1,0].plot(c,FindK(c,eps,e_s,m,f,gamma,opt))
    ax[1,0].set_title('$m =$' + str(m))
    c = 1
    m = np.linspace(1,10,100)
    ax[1,1].plot(m,FindK(c,eps,e_s,m,f,gamma,opt))
    m = 1.25
    gamma = np.linspace(0,1,100)
    ax[0,2].plot(gamma,FindK(c,eps,e_s,m,f,gamma,opt))
    ax[0,2].set_title('$f=$' + str(f))
    gamma = 0.4
    f = np.linspace(0,1,100)
    ax[1,2].plot(f,FindK(c,eps,e_s,m,f,gamma,opt))
    ax[1,2].set_title('$\gamma=$' + str(gamma))
    ax[1,1].set_title('$c=$' + str(c))
    ax[0,0].set_xlabel('$\epsilon$')
    ax[0,1].set_xlabel('$e_s$')
    ax[0,2].set_xlabel('$\gamma$')
    ax[1,0].set_xlabel('$c$')
    ax[1,1].set_xlabel('$m$')
    ax[1,2].set_xlabel('$f$')
    ax[0,0].set_ylabel('$K$')
    ax[0,1].set_ylabel('$K$')
    ax[0,2].set_ylabel('$K$')
    ax[1,0].set_ylabel('$K$')
    ax[1,1].set_ylabel('$K$')
    ax[1,2].set_ylabel('$K$')
    if opt == 'lin': fig.suptitle('$r(p) = 1-\gamma p$')
    else: fig.suptitle('$r(p) = \exp(-\gamma p)$')
    plt.tight_layout()
    plt.show()

def PlotProfDensity(Pn,Ps, r,K):
    ps = np.linspace(0,10,200)
    Ln = Pn.val(ps,r,Ks)
    Ls = Ps.val(ps,r,Ks)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(ps,Ln,label='NS')
    ax.plot(ps,Ls,label='S')
    #ax.plot(ps,Ls-Ln,label='$P_S-P_N$')
    ax.axvline(Ps.pmax,c='blue')
    ax.axhline(0,c='blue')
    idx = np.where(Ln == Ln.max())[0]
    print("p_c: " + str(ps[idx]) + "\tPmax: " + str(Ln.max()))
    idx = np.where(Ls == Ls.max())[0]
    print("p_c: " + str(ps[idx]) + "\tPmax: " + str(Ls.max()))
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylabel('$P$')
    ax.set_xlabel('unit selling price (p)')
    fig.suptitle('$\gamma = $' + str(r.gamma) + ', $K = $' + str(K))
    plt.tight_layout()
    plt.show()

def Case1(Pn,Ps,r):
    '''mu_i = 0 for all i, just solve
    partial_pn Pn = 0
    dPs/dps = 0
    Ps-Pn = 0
    for pn,ps, and K
    '''
    def func(x,Pn,Ps,r):
        return [Pn.grad(x[0],r,x[2]), 
               Ps.grad(x[1],r,x[2]), 
               Ps.val(x[1],r,x[2]) - Pn.val(x[0],r,x[2])]
    Kguess = 1.
    #if Ps.eps > 0.5: Kguess = 3.
    pn,ps,K = fsolve(func, [Pn.pmax/2,Ps.pmax/2,Kguess],args=(Pn,Ps,r)) 
    #print(pn,ps,K,Pn.val(pn,r,K),Ps.val(ps,r,K))
    if Pn.cond(pn,r,K) & Ps.cond(ps,r,K): return pn,ps,K
    else: return -1,-1,-1

def Case2(Pn,Ps,r):
    def func(x,Pn,Ps,r):
        return [Ps.grad(x[0],r,x[1]), 
               Ps.val(x[0],r,x[1]) - Pn.val(Pn.pmax,r,x[1])]
    pn = Pn.pmax
    ps,K = fsolve(func, [Ps.pmax/2,1.],args=(Pn,Ps,r)) 
    if Pn.cond(pn,r,K) & Ps.cond(ps,r,K):
        mu1 = Pn.grad(pn,r,K)
        if mu1>=0: return pn,ps,K
        else: return -1,-1,-1
    else: return -1,-1,-1

def Case3(Pn,Ps,r):
    def func(x,Pn,Ps,r):
        return [Pn.grad(x[0],r,x[1]), 
               Ps.val(Ps.pmax,r,x[1]) - Pn.val(x[0],r,x[1])]
    ps = Ps.pmax
    pn,K = fsolve(func, [Pn.pmax/2,1.],args=(Pn,Ps,r)) 
    if Pn.cond(pn,r,K) & Ps.cond(ps,r,K):
        mu2 = Ps.grad(ps,r,K)
        if mu2>=0: return pn,ps,K
        else: return -1,-1,-1
    else: return -1,-1,-1

def Case4(Pn,Ps,r):
    pn = Pn.pmax
    ps = Ps.pmax
    K = fsolve(lambda x: Ps.val(ps,r,x) - Pn.val(pn,r,x),[1.])[0]
    if Pn.cond(pn,r,K) & Ps.cond(ps,r,K): 
        mu1 = Pn.grad(pn,r,K)
        mu2 = Ps.grad(ps,r,K)
        if (mu1>=0) & (mu2>=0):
            return pn,ps,K
        else: return -1,-1,-1
    else: return -1,-1,-1
   
def FindStuff(args):
    Pn,Ps,r,eps,e_s,n = args
    Ps.eps = eps
    Ps.e_s = e_s
    if n == 1: pn,ps,K = Case1(Pn,Ps,r)
    elif n == 2: pn,ps,K = Case2(Pn,Ps,r)
    elif n == 3: pn,ps,K = Case3(Pn,Ps,r)
    elif n == 4: pn,ps,K = Case4(Pn,Ps,r)
    else: pn,ps,K = -1,-1,-1
    return n, r.gamma, Pn.c, Ps.eps, Ps.m,Ps.f_s, Ps.e_s,Pn.pmax,pn,ps,K, r.val(pn),r.val(ps),Pn.val(pn,r,K),Ps.val(ps,r,K),Ps.val(ps,r,K)-Pn.val(pn,r,K)

def FindAllCases(args):
    '''Find max profit from all cases'''
    Pn,Ps,r,eps,e_s = args

    ns = np.arange(4)+1
    info = np.array([FindStuff((Pn,Ps,r,eps,e_s,n)) for n in ns],dtype=tuple)
    vals = info[:,-1] #(ns) x P  matrix
    idx = np.where(vals==vals.max())[0]
    return info[idx,:][0]

def PoolParty(Pn,Ps,r,num=np.nan):
    df={}
    quantities = ['case','gamma','c','eps','m','f_s','e_s','pmax','pn','ps','K','rn','rs','Pn', 'Ps','Pdiff']

    for i in quantities:
        df[i]=[]

    prefix = 'model6'

    if np.isnan(num) == False: 
        csvname = prefix + "_case" + str(num) + ".csv"
    else:
        csvname = prefix + ".csv"

    tic = time.perf_counter()
    epsilons = np.linspace(0,1,100)
    #sustainability = np.linspace(0,1,10) 
    sustainability = [1-0.158*0.2]
    with multiprocessing.Pool(processes=4) as pool:
        if np.isnan(num) == False: 
            job_args = [(Pn,Ps,r,eps,e_s,num) for eps,e_s in product(epsilons,sustainability)]
            results = pool.map(FindStuff, job_args)
        else:
            job_args = [(Pn,Ps,r,eps,e_s) for eps,e_s in product(epsilons,sustainability)]
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



def Plot4D(size=3,e_s=0.,gamma=1.):
    ''' 
    Generate data for Kmin as a function of c and epsilon, at different values of f and m (3x3 plots)
    Set gamma = 1 (i.e. nondimensionalize the rate / p wrt p_max = 1/gamma for the linear case)
    Set e_s = 0 (impossible but pretend there is such a material that incurs no environmental cost whatsoever -- next step can be to make plots with differing values of e_s)
    0 <= c <= 1 so that 0 <= p <= 1 (since e.g. p_n = 1+c when K=0)
    '''
    #Check that all regions have p_n and p_s <= p_max = 1/gamma, otherwise use the alternative K soln where p_{n,s} = p_max so that P_{N,S} = 0
    cs = np.linspace(0,0.1,50)
    epss = np.linspace(0,1,50)
    ms = np.linspace(0.7,2,size)
    fs = np.linspace(0.1,1,size)
    X,Y = np.meshgrid(cs,epss)
    fig, axs = plt.subplots(size,size,figsize=(8,6))
    for i,j in product(range(size),range(size)):
        Kmins = FindKs(X,Y,e_s,ms[i],fs[j],gamma,opt='lin')
        altKs = (Y-(ms[i]-1)*1/gamma)/(1-e_s)
        pn,ps = FindAnaPs(Kmins,X,Y,e_s,ms[i],fs[j],gamma,opt='lin')
        idxs = np.where((pn<=1/gamma) & (ps<=1/gamma))
        if len(idxs) > 0:
            print(pn[idxs],ps[idxs])
        Z = np.where((pn <= 1/gamma) & (ps <= 1/gamma), Kmins, altKs)
        cf = axs[i,j].contourf(X,Y,Z)
        cbar = fig.colorbar(cf, ax=axs[i,j])
        axs[i,j].set_xlabel('$c$')
        axs[i,j].set_ylabel('$\epsilon$')
        cbar.ax.set_ylabel('$K_{min}$')
        axs[i,j].set_title('$(m,f) = (%.1f, %.1f)$' %(ms[i],fs[j]))
    #fig.suptitle('$K_{min}$')
    plt.tight_layout(pad=1.)
    plt.show()

def FindArrs(df, colnames, fixedqty, fixedval):
    '''Find arrays of parameters at some fixed value of some other quantity (here shown for eta)'''
    #df = pd.read_csv(filename)
    values = [df[name].values for name in colnames]
    etas = df[fixedqty].values 

    #find the nearest index/value of eta in the etas array
    idx = (np.abs(etas-fixedval)).argmin()
    print("Requested: " + fixedqty + " = " + str(fixedval) + "\tfound: " + str(etas[idx]))
    idxs = np.where(etas==etas[idx])
    fixed_vals = [arr[idxs] for arr in values]
    return etas[idx],fixed_vals[0], fixed_vals[1:]

def PlotK(csvname,sus=1-0.158*0.2):
    '''Plot min env tax rate K as a function of epsilon for a given sustainability param e_s (or other variable of your choosing; default is e_s'''
    fig, ax = plt.subplots(2,2,figsize=(6.5,5.5))
    df = pd.read_csv(csvname)
    gamma = df['gamma'].values[0]
    m = df['m'].values[0]
    f = df['f_s'].values[0]
    c = df['c'].values[0]
    pns = df['pn'].values
    pss = df['ps'].values
    pmax = df['pmax'].values[0]
    idxsn = np.where((pns >0) & (pns <pmax))
    idxss = np.where((pss >0) & (pss <pmax))
    print(idxsn)
    print(idxss)


    fixede_s, eps, Ks = FindArrs(df, ['eps','K'],'e_s',sus)
    _,_, arrs = FindArrs(df, ['K','pn','ps','rn','rs','Pn','Ps','case'],'e_s',sus)
    Ks = Ks[0]
    pn,ps,rn,rs,Pn,Ps,ns = arrs
    idxs = np.where(Ks>-1)[0]
    Ks=Ks[idxs]
    eps = eps[idxs]
    pn=pn[idxs]
    ps=ps[idxs]
    rn=rn[idxs]
    rs=rs[idxs]
    Pn=Pn[idxs]
    Ps=Ps[idxs]
    ns = ns[idxs]
    anaK = FindKs(c=c,eps=eps,e_s=fixede_s,m=m,f=f,gamma=gamma)
    #anaKs2 = FindK2(pn,ps,rn,rs,c,eps,fixede_s,m,f)
    
    ax[0,0].plot(eps,Ks,'.',label='num')
    ax[0,0].plot(eps,anaK,'orange',label='ana (-)',zorder=10)
    #ax[0,0].plot(eps,anaKp,'orange',zorder=10)
    ax[0,0].set_xlabel('$\epsilon$')
    ax[0,0].set_ylabel('$K$')
    ax[0,0].legend()
    ax[0,1].set_ylabel('$p$')
    ax[0,1].set_xlabel('$K$')
    anan, anas = FindAnaPs(K=Ks,c=c,eps=eps,e_s=fixede_s,m=m,f=f,gamma=gamma,opt='lin')
    ax[0,1].plot(Ks,pn,'k.',label='$p_n$',zorder=5)
    ax[0,1].plot(Ks,ps,'r.',label='$p_s$',zorder=10)
    ax[0,1].plot(Ks,anan,'*',color='orange',label='ana (N)')
    ax[0,1].plot(Ks,anas,'b*',label='ana (S)')
    ax[0,1].legend()
    #ax[0,1].set_title('$\epsilon=$' + str(fixedeps))
    ax[1,0].plot(eps,Ps-Pn)
    ax[1,0].set_xlabel('$\epsilon$')
    ax[1,0].set_ylabel('$P_{s}-P_n$') 
    #make another subplot showing case number
    ax[1,1].plot(eps,ns,'.') 
    ax[1,1].set_xlabel('$\epsilon$')
    ax[1,1].set_ylabel('Case') 
    plt.tight_layout()
    plt.show()

def PhaseDiagram(csvname):
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
    from fun_model6 import ProfDensityNS, ProfDensityS, LinRate, ExpRate

    gamma = 1.
    c = 0.1
    pmax = 10.
    #rate = ExpRate(gamma)
    rate = LinRate(gamma)
    pmax = 1/gamma
    Pn = ProfDensityNS(c=c,pmax=pmax) 
    Ps = ProfDensityS(c=c,pmax=pmax,s=0.2,eps=0.5) 
    #PlotProfDensity(Pn,Ps, rate,1.)
    PlotProfitVsK_ana(Pn,Ps,rate)
    #Case1(Pn,Ps,rate)
    #csvname = PoolParty(Pn,Ps,rate) 
    #PlotK(csvname)
    #AnalyticK(gamma,c=c,opt='lin')
    #Plot4D(3,e_s=0.,gamma=1.)
