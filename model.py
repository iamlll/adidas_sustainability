import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, curve_fit

def rate(args):
    '''inputs:
        alpha: selling rate parameter for r(p)
        opt = 1: r(p) = alpha/(alpha+p^n)
        opt = 2: r(p) = exp(-alpha/2*p^2)
        opt = 3: r(p) = -alpha*p + 1
        C = unit production cost / avg unit selling price = c/<p>
        F = initial unit environmental cost / avg unit selling price = f_0/<p>
        f = unit environmental cost = a/(r+b) + f_eq, f_eq = 1- a/(1+b)
        
        All variables have been NONDIMENSIONALIZED wrt r_0, <p>!
    '''
    a,b,C,F,alpha,opt,n = args
    
    if opt == 1: #1/p
        r = lambda p: alpha/(alpha+p**n)
        f = lambda p: a/(r(p) + b) + 1 - a/(1+b) #f_eq = 1-a/(1+b), for const env cost set a=0
        func = lambda p: alpha + (1-n)*p**n + n*p**(n-1)*(C + F*f(p) - F*r(p)* a/(r(p) + b)**2)
    elif opt == 2: #gaussian
        r = lambda p: np.exp(-alpha/2*p**2)
        f = lambda p: a/(r(p) + b) + 1 - a/(1+b) #f_eq = 1-a/(1+b), for const env cost set a=0
        func = lambda p: 1-alpha*p*(p- C - F*f(p) + F*r(p)* a/(r(p) + b)**2 )
    elif opt == 3: #linear
        r = lambda p: -alpha*p + 1
        f = lambda p: a/(r(p) + b) + 1 - a/(1+b) #f_eq = 1-a/(1+b), for const env cost set a=0
        func = lambda p: 1-alpha*(2*p- C - F*f(p) + F*r(p)* a/(r(p) + b)**2 )
    else: #exp decay
        r = lambda p: np.exp(-alpha*p)
        f = lambda p: a/(r(p) + b) + 1 - a/(1+b) #f_eq = 1-a/(1+b), for const env cost set a=0
        func = lambda p: 1-alpha*(p- C - F*f(p) + F*r(p)* a/(r(p) + b)**2 )
    guess = [5.] #guess for p given a combination of parameters
    price = fsolve(func, guess)[0]
    ''' 
    ps = np.linspace(-10,10,50)
    ys = func(ps)
    plt.plot(ps,ys)
    plt.axhline(y=0,color='r')
    plt.show()
    '''
    #plot profit = r*(p-C-F*f(r)) vs env cost r*f(r) densities (nondimensionalized)
    #return unit rate, env cost, price
    return opt,alpha,a,b,r(price), f(price), price, r(price)*f(price), r(price)*(price-C-F*f(price))

def PoolParty(c,f0,pavg, case, alpha):
    '''inputs:
        c: constant unit production cost (cost/item)
        f0: initial environmental cost f_0 = f(r=r_0 = r(p=0))
        pavg: average selling price per item 
    '''
    avar = np.linspace(0,10,10)
    bvar = np.linspace(0,10,10)
    df={}
    quantities = ['case','alpha','a','b','r','f','p','rho_E','rho_P']

    for i in quantities:
        df[i]=[]

    tic = time.perf_counter()
    with multiprocessing.Pool(processes=4) as pool:
        job_args = [(a,b,c/pavg,f0/pavg,alpha,case,2) for a,b in product(avar,bvar) if a-b < 1]
        results = pool.map(rate, job_args)
        for res in results:
            for name, val in zip(quantities, res):
                df[name].append(val)

    toc = time.perf_counter()
    print(f"time taken: {toc-tic:0.4f} s, {(toc-tic)/60:0.3f} min")
    data = pd.DataFrame(df)
    pd.set_option("display.max.columns",None)
    return data

def PlotPareto(csvname, case, alphas):
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111)
    df0 = pd.read_csv(csvname) 
    case = df0['case'].values[0]
    print(case)
    for alpha in alphas:
        if case == 1:
            title = '$r_1(p) = \\frac{\\alpha}{\\alpha + p^2}$'
        elif case == 2:
            title = '$r_2(p) = \exp(-\\alpha p^2/2)$'
        elif case == 3:
            title = '$r_3(p) = -\\alpha p + 1$'
        else:
            title = '$r_4(p) = \exp(-\\alpha p)$'
        r_str = '$\\alpha = %.1f$' %(alpha)
        df = df0[df0["alpha"] == alpha]
        rho_P = df['rho_P'].values #nondimensionalized density
        rho_E = df['rho_E'].values #nondim density
        price = df['p'].values
        idx = np.where(price>0)
        profit = rho_P[idx]
        envcost = rho_E[idx]
        ax.plot(profit,envcost,'.', label=r_str)
    ax.set_xlabel('$\\rho_{profit}$')
    ax.set_ylabel('$\\rho_{env}$')
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def FitData(xvals, yvals, varnames, guess=[-1,-3],yerr=[], fit='lin', extrap=[]):
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

    return ans, textstr

if __name__ == "__main__":
    import time
    import pandas as pd
    import multiprocessing
    from itertools import product

    c = 1 #treat unit production cost as const
    f0 = 1 #initial env cost f(r_0) if starting out w/ selling rate of r_0 = r(p=0)
    pavg = 1 #avg unit selling price ($$?)
    case = 4 
    alphas = np.linspace(0.1,1,10)
    csvname = "Case" + str(case) + ".csv"
    dfs = []
    #PoolParty(c,f0,pavg, case, alpha,csvname)
    for alpha in alphas:
        dfs.append( PoolParty(c,f0,pavg, case, alpha))
    df = pd.concat(dfs)
    df.to_csv(csvname, index=False)
    PlotPareto(csvname, case, alphas)


