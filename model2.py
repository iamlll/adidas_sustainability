import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit
from matplotlib.ticker import MaxNLocator, LogLocator
from tools import parse_CSV

def KKT(beta,gamma,pmax):
    ps = np.linspace(0,pmax,100)
    ss = np.linspace(0,1,80)
    P, S = np.meshgrid(ps,ss,sparse=False)
    R = 1-gamma*(1-S)*P
    Rnew = np.where(R>=0, R, -1E-5)
    L = np.where(R>=0, R*(P-(beta+S)), 1E-5)
    
    fig, ax = plt.subplots(figsize=(7,5.5))
    #cf = ax.contourf(P,S,Rnew, levels = MaxNLocator(nbins=20).tick_values(0.,Rnew.max()))
    print(L.min(),L.max())
    cf = ax.contourf(P,S,L, levels = MaxNLocator(nbins=20).tick_values(L.min(),L.max()))
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel('$r$')
    #cbar.ax.set_ylabel('$f = r(p-c)$')

    soln = 1/(gamma*(1-ss))
    s1p = 1/2*(1-beta + np.sqrt((1+beta)**2 -4/gamma))
    p1p = (beta + s1p)/2 + 1/(2*gamma*(1-s1p))
    s1m = 1/6*(5-beta - np.sqrt((1+beta)**2 +12/gamma))
    p1m = (beta + s1m)/2 + 1/(2*gamma*(1-s1m))
    ppoints2 = np.array([p1m,p1p,1./(2*gamma) + beta/2, 1/gamma, pmax, pmax, pmax, pmax]) #cases 1, 2, 4, 5, 9, 11, and 13 correspond to individual points
    spoints2 = np.array([s1m,s1p,0,0,0,1,1-1./(gamma*pmax), 0.5*(1+pmax-beta - 1./(gamma*pmax))])
    f = (-12+(1+beta)*(1+beta+np.sqrt((1+beta)**2+12/gamma))*gamma)**2/(54*gamma*(1+beta+np.sqrt((1+beta)**2+12/gamma)))
    print(f)
    rs = 1-gamma*(1-spoints2)*ppoints2
    fs = rs*(ppoints2-beta-spoints2)
    print(ppoints2)
    print(spoints2)
    print(rs)
    print(fs) 
    idx3 = np.where((ppoints2 <= pmax) & (ppoints2 >= 0) & (spoints2 <= 1) & (spoints2 >= 0))[0]

    idx = np.where(soln <= pmax)[0]
    ax.plot(soln[idx],ss[idx],'.')
    ax.plot(ppoints2[idx3],spoints2[idx3],'go')
    ax.set_xlabel('unit price (p)')
    ax.set_ylabel('sustainability fraction (s)')
    
    plt.tight_layout()
    plt.show()

def betagammaPD(pmax,B,generate=True):
    # 0<B<1
    csvname = "betagammaPD.csv"
    if generate == True:
        #betas = [3.]
        #gammas = [4.]
        gammas = np.linspace(0.1*p_max,2*pmax,50)
        betas = np.linspace(0,2*p_max,50)
        df={}
        quantities = ['pmax','beta','gamma','p','s','r','c','P','E','P/E']

        for i in quantities:
            df[i]=[]

        tic = time.perf_counter()
     
        with multiprocessing.Pool(processes=4) as pool:
            job_args = [(beta,gamma,p_max,B) for beta,gamma in product(betas,gammas)]
            results = pool.map(FindBetaGammaStuff, job_args)
            for res in results:
                for name, val in zip(quantities, res):
                    df[name].append(val)
        toc = time.perf_counter()
        print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
        data = pd.DataFrame(df)
        pd.set_option("display.max.columns",None)
        data.to_csv(csvname, index=False)

    df = pd.read_csv(csvname)
    colnames = ['beta','gamma','P/E']
    bs,gs,Z = parse_CSV(df,colnames)
    fig, ax = plt.subplots(figsize=(7,5.5))
    cf = ax.contourf(bs,gs,Z, levels = MaxNLocator(nbins=20).tick_values(Z.min(),Z.max()))
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel('$P/E$')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('$\gamma$')
    plt.tight_layout()
    plt.show()
    
def FindBetaGammaStuff(args):
    beta, gamma, p_max,B = args
    ns = [1,2,5,9,13] 
     
    info = np.array([etccases(beta,gamma,p_max,n) for n in ns])
    fs = info[:,-1]
    idx = np.where(np.isnan(fs)==False)[0]
    if len(idx) > 0:
        fs = fs[idx]
        ps = info[:,0][idx]
        ss = info[:,1][idx]
        idx2 = np.where(fs==fs.max())[0][0]
        f = fs[idx2]
        p = ps[idx2]
        s = ss[idx2]
        print(p,s,f)
    else:
        f = -1
        p = -1
        s = -1
    r = 1-gamma*(1-s)*p
    c = beta + s
    E = (1-B*s)*r
    return p_max,beta, gamma, p, s,r,c,f,E,f/E

def etccases(beta, gamma,p_max,opt):
    check = True
    s=-1.
    p=-1.
    if opt == 1:
        check = True
        sm = 1/6*(5-beta - np.sqrt((1+beta)**2 + 12/gamma))
        sp = 1/6*(5-beta + np.sqrt((1+beta)**2 + 12/gamma))
        if (1+beta)**2-4/gamma >= 0:
            s2m = 1/2*(1-beta- np.sqrt((1+beta)**2 -4/gamma))
            s2p = 1/2*(1-beta+ np.sqrt((1+beta)**2 -4/gamma))
            ss = np.array([sm,sp,s2m,s2p])
        else:
            ss = np.array([sm,sp])
        idx = np.where((ss>=0) & (ss <=1))[0]
        if len(idx) > 0:    
            s = ss[idx[0]]
        else: s = sm
        p = (beta + s)/2 + 1/(2*gamma*(1-s))
    elif opt == 2:
        mu = 1-0.5*(1+beta*gamma)*(1./(2*gamma) - beta/2 + 1)
        if mu >= 0:
            check = True
            p= 1./(2*gamma) + beta/2
            s=0.
    elif opt == 5: 
        mu5 = 1-gamma*(2*p_max - beta)
        mu1 = 1-gamma*p_max*(p_max - beta + 1)    
        if mu1 >= 0 and mu5 >= 0:
            check = True
            p = p_max
            s=0
    elif opt == 9: 
        mu = gamma*p_max*(p_max - beta - 1) - 1
        if mu >= 0:    
            check = True
            p = p_max
            s=1
    else: #opt = 13 
        mu = 1-gamma/4*(1-p_max + beta + 1./(gamma*p_max))*(3*p_max - beta - 1 + 1./(gamma*p_max))
        if mu >= 0:
            check = True
            p= p_max
            s= 0.5*(1+p_max-beta - 1./(gamma*p_max))
    r = 1-gamma*(1-s)*p
    c = beta + s
    f = r*(p-beta-s)
    cond = check & (p <= p_max) & (p>=0) & (s<=1) & (s>=0)
    #cond = check & (p <= p_max) & (p>=0) & (s<=1) & (s>=0) & (r>=0) & (c>=0)
    if cond == True:
        return p,s,r,c,f
    else: return np.nan, np.nan, np.nan, np.nan, np.nan
    
def PlotKKT(csvname,p_max,case):
    df0 = pd.read_csv(csvname)
    nums = np.unique(df0['case'].values)
    fig, ax = plt.subplots()
    for num in nums:
        df = df0[df0["case"] == num]
        rho_P = df['rho_P'].values #nondimensionalized density
        rho_E = df['rc'].values #nondim density
        price = df['p'].values
        sus = df['s'].values
        betas = df['beta'].values
        gamma = df['beta'].values
        profit = rho_P
        envcost = rho_E
        ax.plot(rho_P,rho_E,'.', label=str(num))
    ax.set_xlabel('profit dens')
    ax.set_ylabel('cost dens')
    ax.legend()
    plt.tight_layout()
    plt.show()

def FindStuff(args):
    beta, gamma, p_max, n = args
    pe,se,re,ce,fe = etccases(beta,gamma,p_max,n)
    #find f vs r*c as analogue to profit density vs env cost density (OK since environmental cost is monotonically decreasing function, wrap into "effective prod cost"). Want to find pareto curves in this scenario.
    #need note about fxnal form of r, c, why they make sense: as s inc, slope of r decreases -> ppl willing to pay more as sustainability goes up
    #want to create curves for each of the cases - solns are mostly points, but want to see how they change with beta and gamma
    #define gammas such that 1/gamma >= p_max (necessary to ensure that the rate r(p,s) >= 0)
    return n,p_max,beta, gamma, pe, se, re,ce, fe,re*ce

def PlotCase10(p_max):
    inv_gammas = np.linspace(p_max,2*p_max,10)
    gs = 1./inv_gammas
    bs = np.linspace(0,2*p_max,10)
    fig, ax = plt.subplots(1,2)
    ss = np.linspace(0,1,50)
    for beta, gamma in product(bs,gs):
        pe,se = case10(beta,gamma,p_max,ss)
        r = 1-gamma*(1-se)*pe
        c = beta + se
        f = r*(pe-c)
        ax[0].plot(pe,se,'.')
        ax[1].plot(f,r*c,'.')
    ax[0].set_ylabel('s')
    ax[0].set_xlabel('p')
    ax[1].set_ylabel('$rc$')
    ax[1].set_xlabel('$f=r(p-c)$')
    plt.tight_layout()
    plt.show()

def PoolParty(p_max,case):
    gs = np.linspace(0.5*p_max,2*p_max,25)
    bs = np.linspace(0,2*p_max,25)
    df={}
    quantities = ['case','pmax','beta','gamma','p','s','r','c','rho_P','rc']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
     
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(beta,gamma,p_max,case) for beta,gamma in product(bs,gs)]
        results = pool.map(FindStuff, job_args)
        for res in results:
            for name, val in zip(quantities, res):
                df[name].append(val)
    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    return data

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product

    #1/gamma >= beta
    csvname = "model2.csv"
    p_max = 10
    KKT(3,4,p_max)
    #PlotKKT(1,1)
    nums = [1]
    #nums = [1,2,5,9,13]
    dfs = []
    '''' 
    for case in nums:
        dfs.append( PoolParty(p_max, case))
    df = pd.concat(dfs)
    df.to_csv(csvname, index=False)
    print(df)
    '''
    B=0.1
    #betagammaPD(p_max,B,generate=True)
    #PlotKKT(csvname, p_max,case)
    
    #PlotCase10(p_max)
    #print(generate_KKT(p_max,10))
