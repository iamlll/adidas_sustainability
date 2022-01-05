import numpy as np
import matplotlib.pyplot as plt

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
    '''Profit density P(p,m) = P_S - P_NS is the difference between market lines offering both sustainable and non-sustainable options. Want this to be >= 0. 
P = r(p) fs ((m-1)p - epsilon +Kn -Ks e_s) where m = sustainability premium that fs(m) % of people are willing to pay.
Production cost of sustainable product is c_s = c_n +epsilon
Kn, Ks taxation rates on carbon emissions (relative importance)

    r(p): selling rate in the absence of any sustainable options, p in units of e(0) = base unit env cost for unsustainable product
    E_s = f_s r(p) e(s)
    E_n = (1-f_s) r(p)
    e(s) = 1-Bs, B=0.158 #unit env cost
    c_s = beta c_0 #c0 = base production cost
    '''
    def __init__(self,K_s=1., K_0=1.2,epsilon=.1,s=0.7,m=1.25,pmax=10):
        self.Kn = K_0
        self.Ks = K_s
        self.eps = epsilon
        self.pmax = pmax #max selling price
        self.s = s #how much of the shoe is made from biodegradable materials - this is a fake number for now
        self.B = 0.158
        self.e_s = 1-self.B*self.s
        self.m = m
        self.f_s = 0.34 #fraction of consumers willing to pay m times the original (NS product) price for a product labeled as sustainable

    def E(self,p,r):
        #env. cost of sustainable product (E_s), env. cost of unsustainable prod (E_n)
        #f = WTP function
        
        Es = self.f_s*r.val(p)*self.e_s
        En = (1-self.f_s)*r.val(p)
        return Es+En

    def val(self,p,r):
        return r.val(p)*self.f_s*((self.m-1)*p-self.eps +self.Kn -self.Ks*self.e_s)

    def grad(self,p,r):
        f = self.f_s
        dPdp = r.grad(p)*f*((self.m-1)*p-self.eps+self.Kn-self.Ks*self.e_s) +r.val(p)*f*(self.m-1)
        return dPdp

    def cond(self,p,r):
        '''check bounds for p and m'''
        check = (p >= 0) & (p <= self.pmax) & (self.val(p,r) >= 0)
        return check

if __name__=="__main__":
    import pandas as pd
    gamma = 0.5
    rate = ExpRate(gamma)
    print(rate.val(1))
    P = ProfDensity(pmax=10) 
    print(P.val(1,rate))
