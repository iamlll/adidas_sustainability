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

class ProfDensityNS:
    '''Profit density of a market selling only nonsustainable products, P_n(p_n) = r(p_n)(p_n-c-K)
    K: tax rate on carbon emissions (measure of sustainability
    r(p): Selling rate of product (e.g. shoes/day)
    c: production cost of NS product
    '''
    def __init__(self,c=1.,pmax=10):
        self.pmax = pmax
        self.c = c

    def val(self,p,r,K):
        return r.val(p)*(p-self.c-K)

    def grad(self,p,r,K):
        gradient = r.grad(p)*(p-self.c-K) +r.val(p)
        return gradient

    def cond(self,p,r,K):
        '''check bounds for p and r; K can be any sign since we're just solving for the minimum required value'''
        check = (p >= 0) & (p <= self.pmax) & (r.val(p) >= 0) 
        return check

class ProfDensityS:
    '''Profit density of a market selling both a NS + sustainable line of products, P_s(p_s) = r(p_s)(m*p_s-(c+eps)-K*e_s)
    K: tax rate on carbon emissions (measure of sustainability
    r(p): Selling rate of product (e.g. shoes/day)
    c: production cost of NS product
    eps: additional production cost of sustainable product
    e_s: tax reduction of sustainable product (as compared to NS product; e_s <= 1)
    '''
    def __init__(self,c=1., eps=.1,s=0.2,m=1.25,pmax=10):
        self.c = c
        self.eps = eps
        self.pmax = pmax #max selling price
        self.e_s = 1-0.158*s
        self.m = m
        self.f_s = 0.34 #fraction of consumers willing to pay m times the original (NS product) price for a product labeled as sustainable
    
    def val(self,p,r,K):
        return r.val(p)*((1-self.f_s)*(p-self.c-K) +self.f_s*(self.m*p-(self.c+self.eps) -K*self.e_s))

    def grad(self,p,r,K):
        gradient = r.grad(p)*((1-self.f_s)*(p-self.c-K) +self.f_s*(self.m*p-(self.c+self.eps) -K*self.e_s)) +r.val(p)*(1+(self.m-1)*self.f_s)
        return gradient

    def cond(self,p,r,K):
        '''check bounds for p and m'''
        check = (p >= 0) & (p <= self.pmax) & (r.val(p) >= 0)
        return check

if __name__=="__main__":
    import pandas as pd
    K = 1.
    gamma = 1
    c = 1.2
    pmax = 10.
    rate = ExpRate(gamma)
    print(rate.val(1))
    Pn = ProfDensityNS(c=c,pmax=pmax) 
    Ps = ProfDensityS(c=c,pmax=pmax) 
    print(Pn.val(1,rate,K))
    print(Ps.val(1,rate,K))
