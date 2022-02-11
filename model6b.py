import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit
from matplotlib.ticker import MaxNLocator, LogLocator
from tools import parse_CSV
import sys

def FindAnaK(c,eps,e_s,m,f,gamma=0.4,opt='lin'):
    if opt == 'lin':
        prefactor = 1/(gamma*(2*e_s-(m+1)+f*(1-e_s)**2))
        K1 = (1+eps*gamma)*(f*(1-e_s)-1) +c*gamma*(m-e_s) +m*f*(e_s-1)
        K2 = np.abs(gamma*(eps +c*(1-e_s)) +e_s-m)*np.sqrt(1+(m-1)*f)
        Kp = prefactor*(K1+K2)
        Km = prefactor*(K1-K2)
        return np.maximum(Kp,Km)
    else: #exponential
        return ((1-m)*c+eps)/(m-e_s) +(f*(1-m)-1)/(f*gamma*(m-e_s)) *np.log(1+f*(m-1))

def FindAnaPs(K,c,eps,e_s,m,f,gamma=0.4,opt='lin'):
    if opt == 'lin':
        pn = 0.5*(1/gamma+c+K)
        ps = 1/(2*gamma)+ (c+K+f*(eps+K*(e_s-1)))*0.5/(1+(m-1)*f)
    else:
        pn = 1/gamma+c+K
        ps = 1/gamma+ (c+K+f*(eps+K*(e_s-1)))/(1+(m-1)*f)
    return pn, ps

def FindK2(pn,ps,rn,rs,c,eps,e_s,m,f):
    '''min K given by K > [r(pn)/r(ps) (pn-c) -(ps-c) +f(ps(1-m)+c)] / [f(1-es) -1 + r(pn)/r(ps)]
    Basically, minimize profits just wrt ps, pn for a given K, then find K_min from the above expression and compare whether K>K_min. If yes, great, if not, invalid solution.
    '''
    numer = rn/rs *(pn-c)-(ps-c) + f*(ps*(1-m)+eps)
    denom = f*(1-e_s) - 1 + rn/rs
    return numer/denom

def FindK3(c,eps,e_s,m,f,gamma=0.4,opt='lin'):
    if opt == 'lin':
        prefactor = 1/(gamma*(2*e_s-(m+1)+f*(1-e_s)**2))
        K1 = (1+f*(m-1))*(e_s-1) -gamma*(eps*(1+f*(e_s-1)) +c*(e_s-m))
        K2 = np.abs(gamma*(eps +c*(1-e_s)) +e_s-m)*np.sqrt(1+(m-1)*f)
        Kp = prefactor*(K1+K2)
        Km = prefactor*(K1-K2)
        return np.maximum(Kp,Km)
    else: #exponential
        return ((1-m)*c+eps)/(m-e_s) +(f*(1-m)-1)/(f*gamma*(m-e_s)) *np.log(1+f*(m-1))

def AnalyticK(gamma=0.4,e_s=1-0.158*0.2,m=1.25,c=1,f=0.34,opt='lin'):
    eps = np.linspace(0,1,100)
    fig, ax = plt.subplots(2,3,figsize=(7,5.5))
    ax[0,0].plot(eps,FindK3(c,eps,e_s,m,f,gamma,opt=opt))
    ax[0,0].set_title('$e_s = 1-0.158*0.2$')
    e_s = np.linspace(0,1,100) 
    eps = 0.2
    ax[0,1].plot(e_s,FindK3(c,eps,e_s,m,f,gamma,opt))
    ax[0,1].set_title('$\epsilon =$' + str(eps))
    e_s = 1-0.158*0.2
    c = np.linspace(0,5,500)
    ax[1,0].plot(c,FindK3(c,eps,e_s,m,f,gamma,opt))
    ax[1,0].set_title('$m =$' + str(m))
    c = 1
    m = np.linspace(1,10,100)
    ax[1,1].plot(m,FindK3(c,eps,e_s,m,f,gamma,opt))
    m = 1.25
    gamma = np.linspace(0,1,100)
    ax[0,2].plot(gamma,FindK3(c,eps,e_s,m,f,gamma,opt))
    ax[0,2].set_title('$f=$' + str(f))
    gamma = 0.4
    f = np.linspace(0,1,100)
    ax[1,2].plot(f,FindK3(c,eps,e_s,m,f,gamma,opt))
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

def Case1(Pn,Ps,r,K):
    '''mu_i = 0 for all i, just solve
    partial_pn Pn = 0
    dPs/dps = 0
    Ps-Pn = 0
    for pn,ps, and K
    '''
    pn = fsolve(lambda x: Pn.grad(x,r,K), [Pn.pmax/2])[0]
    ps = fsolve(lambda x: Ps.grad(x,r,K), [Ps.pmax/2])[0]
    print(pn,ps,K,Pn.val(pn,r,K),Ps.val(ps,r,K))

    if Pn.cond(pn,r) & Ps.cond(ps,r): return pn,ps
    else: return -1,-1

def Case2(Pn,Ps,r,K):
    pn = Pn.pmax
    ps = fsolve(lambda x: Ps.grad(x,r,K), [Pn.pmax/2])[0]
    if Pn.cond(pn,r) & Ps.cond(ps,r):
        mu1 = Pn.grad(pn,r,K)
        if mu1>=0: return pn,ps
        else: return -1,-1
    else: return -1,-1

def Case3(Pn,Ps,r,K):
    ps = Ps.pmax
    pn = fsolve(lambda x: Pn.grad(x,r,K), [Pn.pmax/2])[0]
    if Pn.cond(pn,r) & Ps.cond(ps,r):
        mu2 = Ps.grad(ps,r,K)
        if mu2>=0: return pn,ps
        else: return -1,-1
    else: return -1,-1

def Case4(Pn,Ps,r,K):
    pn = Pn.pmax
    ps = Ps.pmax
    if Pn.cond(pn,r) & Ps.cond(ps,r): 
        mu1 = Pn.grad(pn,r,K)
        mu2 = Ps.grad(ps,r,K)
        if (mu1>=0) & (mu2>=0):
            return pn,ps
        else: return -1,-1
    else: return -1,-1
   
def FindStuff(args):
    Pn,Ps,r,eps,K,n = args
    Ps.eps = eps
    if n == 1: pn,ps = Case1(Pn,Ps,r,K)
    elif n == 2: pn,ps = Case2(Pn,Ps,r,K)
    elif n == 3: pn,ps = Case3(Pn,Ps,r,K)
    elif n == 4: pn,ps = Case4(Pn,Ps,r,K)
    else: pn,ps,K = -1,-1
    return n, r.gamma, Pn.c, Ps.eps, Ps.m,Ps.f_s, Ps.e_s,Pn.pmax,pn,ps,K, r.val(pn),r.val(ps),Pn.val(pn,r,K),Ps.val(ps,r,K), Ps.val(ps,r,K)-Pn.val(pn,r,K)

def FindAllCases(args):
    '''Find max profit from all cases'''
    Pn,Ps,r,eps,K = args

    ns = np.arange(4)+1
    info = np.array([FindStuff((Pn,Ps,r,eps,K,n)) for n in ns],dtype=tuple)
    vals = info[:,-1] #(ns) x P  matrix
    idx = np.where(vals==vals.max())[0]
    return info[idx,:][0]

def PoolParty(Pn,Ps,r,num=np.nan):
    df={}
    quantities = ['case','gamma','c','eps','m','f_s','e_s','pmax','pn','ps','K','rn','rs','Pn', 'Ps','Pdiff']

    for i in quantities:
        df[i]=[]

    prefix = 'model6b'

    if np.isnan(num) == False: 
        csvname = prefix + "_case" + str(num) + ".csv"
    else:
        csvname = prefix + ".csv"

    tic = time.perf_counter()
    Ks = np.linspace(0,2,100)
    epsilons = np.linspace(0,1,80)
    with multiprocessing.Pool(processes=4) as pool:
        if np.isnan(num) == False: 
            job_args = [(Pn,Ps,r,eps,K,num) for K,eps in product(Ks,epsilons)]
            results = pool.map(FindStuff, job_args)
        else:
            job_args = [(Pn,Ps,r,eps,K) for K,eps in product(Ks,epsilons)]
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

def PlotK(csvname):
    '''Plot min env tax rate K as a function of epsilon for a given sustainability param e_s (or other variable of your choosing; default is e_s'''
    def FindK2(pn,ps,r,c,eps,e_s,m,f):
        '''min K given by K > [r(pn)/r(ps) (pn-c) -(ps-c) +f(ps(1-m)+c)] / [f(1-es) -1 + r(pn)/r(ps)]
        Basically, minimize profits just wrt ps, pn for a given K, then find K_min from the above expression and compare whether K>K_min. If yes, great, if not, invalid solution.
        '''
        numer = r.val(pn)/r.val(ps)*(pn-c)-(ps-c) + f*(ps*(1-m)+eps)
        denom = f*(1-e_s) - 1 + r.val(pn)/r.val(ps)
        return numer/denom

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    df = pd.read_csv(csvname)
    pns = df['pn'].values
    pss = df['ps'].values
    pmax = df['pmax'].values[0]
    idxsn = np.where((pns >0) & (pns <pmax))
    idxss = np.where((pss >0) & (pss <pmax))
    print(idxsn)
    print(idxss)
    e_s = df['e_s'].values[0]
    fixedval, eps, Ks = FindArrs(df, ['eps','K'],'e_s',sus)
    ax.plot(eps,Ks[0])
    ax.set_xlabel('$\epsilon$')
    ax.set_ylabel('$K$')
    plt.tight_layout()
    plt.show()

def FitData(xvals, yvals, varnames, guess=[1,1],yerr=[],extrap=[]):
    def fitlinear(x,a,b):
        f = a*x + b 
        return f

    bnds = ([-10,-10],[10,10]) #bounds for weak coupling fit
    if len(yerr) > 0:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, sigma=yerr, p0=guess,bounds=bnds)
    else:
        param, p_cov = curve_fit(fitlinear,xvals, yvals, p0=guess,bounds=bnds)
    #print(param)
    a,b = param
    X,Y = varnames
    aerr, berr = np.sqrt(np.diag(p_cov)) #standard deviation of the parameters in the fit
    
    if len(extrap) > 0:
        ans = np.array([fitlinear(x,a,b) for x in extrap])
    else:    
        ans = np.array([fitlinear(x,a,b) for x in xvals])
    
    textstr = '\n'.join((
        r'$%s(%s) = a%s + b$' % (Y, X, X),
        r'$a=%.4f \pm %.4f$' % (a, aerr),
        r'$b=%.4f \pm %.4f$' % (b, berr)
        ))
    print(textstr)
    return ans, textstr

def PhaseDiagram(csvname):
    import numpy.ma as ma
    fig, ax = plt.subplots(1,3,figsize=(12,5))
    df = pd.read_csv(csvname)
    
    Ks = df['K'].values
    eps = df['eps'].values
    pns = df['pn'].values
    pss = df['ps'].values
    rns = df['rn'].values
    rss = df['rs'].values
    c = df['c'].values[0]
    e_s = df['e_s'].values[0]
    gamma = df['gamma'].values[0]
    m = df['m'].values[0]
    f = df['f_s'].values[0]
    ns = df['case'].values
    colnames = ['K','eps','Pdiff']
    colnames3 = ['K','eps','case']
    X,Y,Z = parse_CSV(df,colnames)
    #Zcase = np.reshape(ns,Z.shape)
    Zcase = df.pivot_table(index='K', columns='eps', values='case').T.values
    cpmin = 0.
    cf = ax[0].contourf(X,Y,Z, levels = MaxNLocator(nbins=20).tick_values(cpmin,Z.max()))

    #Find minimum K required to get Ps > Pn as from eq (284) -- this clearly doesn't work, since the resulting spectrum of K's that satisfy K>Kmin does not cover the whole area of the phase diagram where (Ps-Pn)>0.
    Kmins = FindAnaK(c,Y,e_s,m,f,gamma,opt='lin')
    Kmins2 = FindK2(pns,pss,rns,rss,c,eps,e_s,m,f)
    #at each value of epsilon
    Kmesh2,_ = np.meshgrid(np.unique(Kmins2),np.unique(eps))
    print(Kmesh2.shape,X.shape)
    Kmins3 = FindK3(c,Y,e_s,m,f,gamma,opt='lin')
    Lx = np.where((X>=Kmins), X, np.nan)
    Ly = np.where((X>=Kmins), Y, np.nan)
    maskedLx = ma.masked_invalid(Lx)
    maskedLy = ma.masked_invalid(Ly)
    ax[0].plot(maskedLx,maskedLy,'b.')

    fixedeps, Xlin,pnfix = FindArrs(df, ['K','pn'],'eps',0.1)
    fixedeps, Xlin,psfix = FindArrs(df, ['K','ps'],'eps',0.1)
    ax[1].plot(Xlin,pnfix[0],label='N')
    ax[1].plot(Xlin,psfix[0],label='S')
    fitn, fits = FindAnaPs(K=Xlin,c=c,eps=fixedeps,e_s=e_s,m=m,f=f,gamma=gamma,opt='lin')
    ax[1].plot(Xlin,fitn,label='ana (N)')
    ax[1].plot(Xlin,fits,label='ana (S)')
    #print((pnfix[0]-fitn)[:10])
    #print((psfix[0]-fits)[:10])
    ax[1].set_title('$\epsilon=0.1$')
    ax[1].set_xlabel('$K$')
    ax[1].set_ylabel('$p$')
    ax[1].legend()
    #ax[1].set_ylabel('$P_S-P_N$')
    
    nbins_case = np.unique(Zcase)
    print(nbins_case)
    nbins_case = int(nbins_case.max()-nbins_case.min())
    cf3 = ax[2].contourf(X,Y,Zcase, levels = MaxNLocator(nbins=nbins_case+1).tick_values(Zcase.min(),Zcase.max()))
    cbar = fig.colorbar(cf, ax=ax[0])
    #cbar2 = fig.colorbar(cf2, ax=ax[1])
    cbar3 = fig.colorbar(cf3, ax=ax[2])
    fig.suptitle("$\gamma= $" + str(gamma) + ", $m= $" + str(m) + ", $f_s =" + str(f) + "$, $c=" + str(c) + "$")
    cbar.ax.set_ylabel('$P_S-P_N$')
    #cbar2.ax.set_ylabel(colnames2[-1])
    cbar3.ax.set_ylabel(colnames3[-1])

    ax[0].set_xlabel('$K$')
    ax[0].set_ylabel('$\epsilon$')
    ax[2].set_xlabel('$K$')
    ax[2].set_ylabel('$\epsilon$')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product
    from fun_model6b import ProfDensityNS, ProfDensityS, LinRate, ExpRate

    gamma = 1.
    c = 0.01
    pmax = 10.
    #rate = ExpRate(gamma)
    rate = LinRate(gamma)
    pmax = 1/gamma
    Pn = ProfDensityNS(c=c,pmax=pmax) 
    Ps = ProfDensityS(c=c,pmax=pmax,s=0.2,eps=0.0) 
    print(Ps.eps)
    K = 1.0889974638602358
    #PlotProfDensity(Pn,Ps,rate)
    #Case1(Pn,Ps,rate,K)

    csvname = PoolParty(Pn,Ps,rate) 
    PlotK(csvname)
    #PhaseDiagram(csvname)
    #AnalyticK(gamma,c=c,opt='lin')
