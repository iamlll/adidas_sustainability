import numpy as np
import matplotlib.pyplot as plt

class WTP_linear:
    #f(1.25) = 0.34, f(1) = 1
    def __init__(self):
        self.slope = -2.64
        self.b = self.slope+1 #want f(1) = 1
        self.mmax = self.b/self.slope
    def val(self,m):
        if (m < 1): return 1.
        elif m >= self.b/self.slope:
            return 0.
        else: 
            return self.b + self.slope*m
    def grad(self,m):
        #gradient wrt m
        if (m >= 1) & (m <= self.b/self.slope):
            return self.slope
        else: 
            return 0.
    def lap(self,m):
        #laplacian (i.e. 2nd deriv)
        return 0.
    def jerk(self,m):
        #3rd derivative
        return 0.

class WTP_cubic:
    '''
    f_s(m) = 1/(1+am^3), a = 1 #regulates "willingness-to-pay" factor; for a=1, f_s(1.25) = 0.34 means that around 34% of consumers are willing to pay 1.25x the initial price of a product for a more sustainably produced version
    '''
    def __init__(self,mmax=100):
        self.a = 1.
        self.mmax = mmax

    def val(self,m):
        #willingness to pay for a more sustainable product, f_s(m) = 1/1(1+am^3)
        return 1./(1+self.a*m**3)
 
    def grad(self,m):
        #gradient wrt m
        return -3*m**2*self.a /(1+self.a*m**3)**2

    def lap(self,m):
        #laplacian (i.e. 2nd deriv)
        return 6*self.a*m*(2*self.a*m**3-1)/(1+self.a*m**3)**3

    def jerk(self,m):
        #3rd derivative
        return -6*self.a/(1+self.a*m**3)**4 *(10*self.a**2*m**6 -16*self.a*m**3 +1)

class LinRate():
    '''r(p): selling rate / day of a given product'''

    def __init__(self,gamma=1):
        self.gamma = gamma

    def val(self,p):
        #willingness to pay for a more sustainable product, f_s(m) = 1/1(1+am^3)
        return 1-self.gamma*p
 
    def grad(self,p):
        #gradient wrt m
        return -self.gamma

    def lap(self,p):
        #laplacian (i.e. 2nd deriv)
        return 0.

    def jerk(self,p):
        #3rd derivative
        return 0.
   
class ExpRate():
    def __init__(self,gamma=1):
        self.gamma = gamma

    def val(self,p):
        #willingness to pay for a more sustainable product, f_s(m) = 1/1(1+am^3)
        return np.exp(-self.gamma*p)
 
    def grad(self,p):
        #gradient wrt m
        return -self.gamma*self.val(p)

    def lap(self,p):
        #laplacian (i.e. 2nd deriv)
        return self.gamma**2*self.val(p)

    def jerk(self,p):
        #3rd derivative
        return -self.gamma**3* self.val(p)

class ProfDensity:
    '''Profit density P(p,m) = rs(ps-cs) + rn(p0 -cn) as function of unit selling price p0 (ps = m*p0) and willingness to pay (WTP) factor m
    gamma, s, beta_i, a ideally known from data; optimize wrt m (sustainability premium) and p_0 (init selling price in absence of sustainable options)
    Make phase plots wrt K_n, K_s (regulations for sustainable and nonsustainable products)    

    r(p): selling rate in the absence of any sustainable options, p in units of e(0) = base unit env cost for unsustainable product
    E_s = f_s r(p) e(s)
    E_n = (1-f_s) r(p)
    e(s) = 1-Bs, B=0.158 #unit env cost
    c_s = beta c_0 #c0 = base production cost
    '''
    def __init__(self,K_s=1., K_0=1.2,beta=1.7,s=0.7,c0=0.5,pmax=10):
        self.Kn = K_0
        self.Ks = K_s
        self.beta = beta
        self.pmax = pmax #max selling price
        self.s = s #how much of the shoe is made from biodegradable materials - this is a fake number for now
        self.B = 0.158
        self.c0 = c0
        self.e_s = 1-self.B*self.s

    def E(self,p,m,f,r):
        #env. cost of sustainable product (E_s), env. cost of unsustainable prod (E_n)
        #f = WTP function
        
        Es = f.val(m)*r.val(p)*self.e_s
        En = (1-f.val(m))*r.val(p)
        return Es, En, Es+En

    def cost(self):
        #c_s and c_n
        return self.beta*self.c0, self.c0

    def val(self,p,m,f,r):
        cs,cn = self.cost() 
        Es, En,_ = self.E(p,m,f,r)
        term1 = f.val(m)*r.val(p)*(m*p-cs) -self.Ks*Es  
        term2 = (1-f.val(m))*r.val(p)*(p-cn) - self.Kn*En
        return term1 + term2

    def normval(self,p,m,f,r):
        P = self.val(p,m,f,r)
        Etot = self.E(p,m,f,r)[-1]
        if Etot == 0: return -1
        else: return P/Etot

    def grad(self,p,m,wtp,r):
        f = wtp.val(m)
        dPdp = r.grad(p)*(p-self.c0-self.Kn +((m-1)*p-(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn)*f) + r.val(p)*(1+(m-1)*f)
        dPdm = r.val(p)*(wtp.grad(p)*((m-1)*p-(self.beta-1)*self.c0-self.Ks*self.e_s +self.Kn) +p*f)
        return dPdp, dPdm

    def hess(self,p,m,wtp,r):
        f = wtp.val(m)
        gf = wtp.grad(m)
        lf = wtp.lap(m)
        d2pP = r.lap(p)*(p-self.c0-self.Kn+((m-1)*p-(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn)*f) +2*r.grad(p)*((m-1)*f+1)
        dpdmP = r.grad(p)*(p*f+gf*((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn)) +r.val(p)*(f+(m-1)*gf)
        d2mP = r.val(p)*(lf*((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn) +2*p*gf)
        H = np.array([[d2pP,dpdmP],[dpdmP,d2mP]])
        return H

    def GetConstraint(self,p,m,wtp,r):
        '''returns concavity condition to ensure that Hessian is negative semidefinite (i.e. that P= objective function is always maximal)
        '''
        H = self.hess(p,m,wtp,r)
        x = H[0,0] + H[1,1]
        q = H[0,1]**2 -H[0,0]*H[1,1]
        return x,q

    def GradConstraint(self,p,m,wtp,r):
        '''returns gradient of convexity constraint'''
        H = self.hess(p,m,wtp,r)
        f = wtp.val(m)
        gf = wtp.grad(m)
        lf = wtp.lap(m)
        
        dpH11 = r.jerk(p)*(p-self.c0 -self.Kn +((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn)*f) +3*r.lap(p)*(1+(m-1)*f)
        dmH11 = r.lap(p)*(p*f +gf*((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn)) +2*r.grad(p)*(f +(m-1)*gf) #also = dpH12
        dmH12 = r.grad(p)*(p*gf +lf*((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn)) +r.val(p)*(2*gf +(m-1)*lf) #also = dpH22
        dmH22 = r.val(p)*(3*p*lf + wtp.jerk(m)*((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn))
        grad11 = np.array([dpH11, dmH11])
        grad12 = np.array([dmH11, dmH12])
        grad22 = np.array([dmH12, dmH22])
        gradx = grad11 + grad22
        gradq = 2*H[0,1]* grad12 - H[1,1]*grad11 - H[0,0]*grad22
        return gradx, gradq

    def cond(self,p,m,f):
        '''check bounds for p and m'''
        check = (p >= 0) & (p <= self.pmax) & (m >= 0) & (m <= f.mmax) 
        return check

if __name__=="__main__":
    import pandas as pd
    fs = WTP_cubic()
    gamma = 0.5
    rate = ExpRate(gamma)
    print(rate.val(1))
    P = ProfDensity(pmax=10) 
    #print(P.GetConstraint(1,1.25,fs))
    print(P.val(1,1.25,fs,rate))
    #print(P.normval(1,1.25,fs))
    #print(P.hess(2,1.25,fs))
