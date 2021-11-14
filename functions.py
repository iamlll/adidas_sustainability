import numpy as np
import matplotlib.pyplot as plt

class ProfDensity:
    '''Profit density f(p,s) = r(p,s) (p-c(s)) as function of unit selling price and sustainability fraction s
    r(p,s) = 1-gamma*(1-alpha*s)*p
    c(s) = beta + s
    '''
    def __init__(self,beta=1.,gamma=1.,alpha=0.5,pmax=10.):
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.pmax = pmax #max selling price
        self.B = np.nan

    def GetRate(self,p,s):
        return 1-self.gamma*(1-self.alpha * s)*p

    def GetCost(self,s):
        return s + self.beta

    def Value(self,p,s):
        return self.GetRate(p,s)*(p-self.GetCost(s))
 
    def GetEnvCost(self,p,s):
        return self.GetRate(p,s)*self.GetCost(s)

    def Gradient(self,p,s):
        gradp = 1-self.gamma*(1-self.alpha*s)*(2*p-s-self.beta)
        grads = self.gamma*self.alpha*p*(p-2*s-self.beta)+self.gamma-1
        return np.array([gradp,grads])

    def GetConstraint(self,p,s):
        '''returns concavity condition g_7 (i.e. ensures that Hessian is negative semidefinite)
        '''
        g7 = (self.alpha*(2*p-self.beta-2*s)+1)**2-4*(1-self.alpha*s)*self.alpha*p
        return g7

    def GradConstraint(self,p,s):
        '''returns gradient of g_7'''
        gradp = 4*self.alpha**2 * (2*p-self.beta-s)
        grads = -4*self.alpha*(self.alpha*(p-self.beta-2*s)+1)
        return np.array([gradp,grads])

    def Hessian(self,p,s):
        mixed = self.gamma*self.alpha*( 2*p-self.beta-2*s) + self.gamma
        return np.array([[-2*self.gamma*(1-self.alpha*s), mixed],[mixed, -2*self.gamma*self.alpha*p]])
    
    def cond(self,p,s):
        '''check bounds for p and s'''
        check = (p >= 0) & (p <= self.pmax) & (s >= 0) & (s <= 1) & (self.GetRate(p,s) >= 0)
        return check

class ProfEnvDensity:
    '''Profit density including environmental cost density f(p,s) - E(p,s) = r(p,s) (p-c(s) - e(p,s)) as function of unit selling price and sustainability fraction s
    r(p,s) = 1-gamma*(1-alpha*s)*p --> unit selling rate (shoes/time)
    c(s) = beta + s --> unit prod/selling cost
    e(p,s) = 1-Bs --> unit environmental cost
    '''
    def __init__(self,gamma=1.,beta=1., alpha=0.5,B=0.5,pmax=10.):
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.B = B
        self.pmax=pmax

    def GetRate(self,p,s):
        return 1-self.gamma*(1-self.alpha * s)*p

    def GetCost(self,s):
        return s + self.beta

    def GetEnvCost(self,p,s):
        return self.GetRate(p,s)*(1-self.B*s)

    def Value(self,p,s):
        return self.GetRate(p,s)*(p-self.GetCost(s)) - self.GetEnvCost(p,s)

    def Gradient(self,p,s):
        gradp = 1-self.gamma*(1-self.alpha*s)*(2*p-s-self.beta)
        grads = self.gamma*self.alpha*p*(p-2*s-self.beta)+self.gamma-1
        Egradp = -self.gamma*(1-self.alpha*s)*(1-self.B*s)
        Egrads = self.gamma*(1+self.B-2*self.B*s)*p-self.B
        return np.array([gradp-Egradp,grads-Egrads])

    def GetConstraint(self,p,s):
        '''returns concavity condition g_7' (i.e. ensures that Hessian is negative semidefinite)
        '''
        g7 = (self.alpha*(2*p-self.beta-2*(1-self.B)*s-1)+1- self.B)**2-4*(1-self.B)*(1-self.alpha*s)*self.alpha*p
        return g7

    def GradConstraint(self,p,s):
        '''returns gradient of g_7'''
        gradp = 4*self.alpha**2 * (2*p-(1-self.B)*s-self.beta-1)
        grads = 4*self.alpha*( -(1-self.B)**2 + self.alpha*(1-self.B)*(1+self.beta + 2*(1-self.B)*s - p))
        return np.array([gradp,grads])

    def Hessian(self,p,s):
        mixed = self.gamma*(self.alpha*( 2*p-2*(1-self.B)*s- self.beta-1) + 1-self.B)
        return np.array([[-2*self.gamma*(1-self.alpha*s), mixed],[mixed, -2*self.gamma*self.alpha*(1-self.B)*p]])

    def cond(self,p,s):
        '''check bounds for p and s'''
        check = (p >= 0) & (p <= self.pmax) & (s >= 0) & (s <= 1) & (self.GetRate(p,s) >= 0)
        return check

if __name__=="__main__":
    import pandas as pd
    p,s=np.random.uniform(size=2) #(p,s)
    print(p,s)
    f1 = ProfEnvDensity()
    print(f1.Value(p,s))
    print(f1.GetConstraint(p,s))
    print(f1.Hessian(p,s))
    
