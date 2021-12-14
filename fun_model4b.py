import numpy as np
import matplotlib.pyplot as plt

class WTP_linear:
    #f(1.25) = 0.34, f(1) = 1
    def __init__(self):
        self.slope = -2.64
        self.b = self.slope+1 #want f(1) = 1
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
    def __init__(self):
        self.a = 1.

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

class ProfDensity:
    '''Profit density P(p,m) = rs(ps-cs) + rn(p0 -cn) as function of unit selling price p0 (ps = m*p0) and willingness to pay (WTP) factor m
    gamma, s, beta_i, a ideally known from data; optimize wrt m (sustainability premium) and p_0 (init selling price in absence of sustainable options)
    Make phase plots wrt K_n, K_s (regulations for sustainable and nonsustainable products)    

    r(p) = 1-\gamma p #selling rate in the absence of any sustainable options, p in units of e(0) = base unit env cost for unsustainable product
    E_s = f_s r(p) e(s)
    E_n = (1-f_s) r(p)
    e(s) = 1-Bs, B=0.158 #unit env cost
    c_s = beta c_0 #c0 = base production cost
    '''
    def __init__(self,K_s=1., K_0=1.2,beta=1.7,gamma=1.,s=0.7,c0=0.5):
        self.Kn = K_0
        self.Ks = K_s
        self.beta = beta
        self.gamma = gamma
        self.pmax = 1./gamma #max selling price
        self.s = s #how much of the shoe is made from biodegradable materials - this is a fake number for now
        self.B = 0.158
        self.c0 = c0
        self.e_s = 1-self.B*self.s

    def rate(self,p): #selling rate given no sustainable product (s=0)
        return 1-self.gamma*p

    def E(self,p,m, f):
        #env. cost of sustainable product (E_s), env. cost of unsustainable prod (E_n)
        #f = WTP function
        Es = f.val(m)*self.rate(p)*self.e_s
        En = (1-f.val(m))*self.rate(p)
        Etot = Es + En
        return Es, En, Etot

    def cost(self,p,m,wtp):
        #c_s and c_n
        return self.beta*self.c0, self.c0

    def val(self,p,m,f):
        cs,cn = self.cost(p,m,f) 
        Es, En,_ = self.E(p,m,f)
        return f.val(m)*self.rate(p)*(m*p-cs) -self.Ks*Es  + (1-f.val(m))*(p-cn) - self.Kn*En

    def normval(self,p,m,f):
        P = self.val(p,m,f)
        Etot = self.E(p,m,f)[-1]
        if Etot == 0: return -1
        else: return P/Etot

    def grad(self,p,m,wtp):
        f = wtp.val(m)
        dPdp = (1-2*self.gamma*p)*(m-1)*f + 1 + self.gamma*((self.beta-1)*self.c0 + self.Ks*self.e_s - self.Kn)*f
        dPdm = (1-self.gamma*p)*(wtp.grad(m)*((m-1)*p - (self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn) +p*f)
        return dPdp, dPdm

    def hess(self,p,m,wtp):
        f = wtp.val(m)
        gf = wtp.grad(m)
        lf = wtp.lap(m)
        r = self.rate(p)
        d2pP = -2*self.gamma*(m-1)*f
        dpdmP = (1-2*self.gamma*p)*(f+(m-1)*gf) +self.gamma*gf*((self.beta-1)*self.c0 + self.Ks*self.e_s -self.Kn)
        d2mP = (1-self.gamma*p)*lf*((m-1)*p -(self.beta-1)*self.c0 -self.Ks*self.e_s +self.Kn) +2*p*(1-self.gamma*p)*gf
        H = np.array([[d2pP,dpdmP],[dpdmP,d2mP]])
        #print(np.linalg.eig(H)[0])
        return H

    def GetConstraint(self,p,m,wtp):
        '''returns concavity condition to ensure that Hessian is negative semidefinite (i.e. that P= objective function is always maximal)
        '''
        H = self.hess(p,m,wtp)
        x = H[0,0] + H[1,1]
        q = H[0,1]**2 -H[0,0]*H[1,1]
        return x,q

    def GradConstraint(self,p,m,wtp):
        '''returns gradient of convexity constraint'''
        H = self.hess(p,m,wtp)
        f = wtp.val(m)
        gf = wtp.grad(m)
        lf = wtp.lap(m)
        
        r = self.rate(p)
        value = -2*self.gamma*(f + (m-1)*gf)
        value2 = (1-2*self.gamma*p)*(2*gf + (m-1)*lf) + self.gamma*lf*((self.beta-1)*self.c0 + self.Ks*self.e_s -self.Kn)
        grad11 = np.array([0.,value])
        grad12 = np.array([value, value2])
        grad22 = np.array([value2, (1-self.gamma*p)*((m-1)*p + self.Kn -(self.beta-1)*self.c0 -self.Ks*self.e_s)* wtp.jerk(m) + 3*p*(1-self.gamma*p)*lf])
        gradx = grad11 + grad22
        gradq = 2*H[0,1]* grad12 - H[1,1]*grad11 - H[0,0]*grad22
        return gradx, gradq

    def cond(self,p,m,f):
        '''check bounds for p and m'''
        check = (p >= 0) & (p <= self.pmax) & (m >= 0) 
        return check

if __name__=="__main__":
    import pandas as pd
    fs = WTP_cubic()
    gamma = 0.5
    P = ProfDensity(gamma=gamma) 
    print(P.GetConstraint(1,1.25,fs))
    print(P.val(1,1.25,fs))
    print(P.normval(1,1.25,fs))
    #print(P.hess(2,1.25,fs))
