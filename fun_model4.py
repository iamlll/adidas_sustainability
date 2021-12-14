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

    r(p,s) = 1-gamma*(1-alpha*s)*p
    r_0(p0) = 1-\gamma p0 #selling rate in the absence of any sustainable options
    E_s = f_s r_0 e(s)
    E_n = (1-f_s) r_0 e(0)
    e(s) = 1-Bs, B=0.158 #unit env cost
    c_s(E_s) = beta_s + K_s E_s
    c_n(E_n) = beta_n + K_n E_0
    f_s(m) = 1/(1+am^3), a = 1 #regulates "willingness-to-pay" factor; for a=1, f_s(1.25) = 0.34 means that around 34% of consumers are willing to pay 1.25x the initial price of a product for a more sustainably produced version
    '''
    def __init__(self,Ks=1.5, K0=1.,beta_s=0.2,beta_n=0.1,gamma=1.,s=0.7):
        #alpha_i's should be <= 1
        self.gamma = gamma
        self.beta = np.array([beta_s,beta_n])
        self.K = np.array([Ks,K0]) 
        self.pmax = 1./self.gamma #max selling price
        self.s = s #how much of the shoe is made from biodegradable materials - this is a fake number for now
        self.B = 0.158
        self.e_s = 1-self.B*self.s

    def r0(self,p):
        return 1-self.gamma*p

    def E(self,p,m, f):
        #env. cost of sustainable product (E_s), env. cost of unsustainable prod (E_n)
        #f = WTP function
        return f.val(m)*self.r0(p)*self.e_s, (1-f.val(m))*self.r0(p)

    def cost(self,p,m,wtp):
        #c_s and c_n
        return self.beta + self.K*self.E(p,m,wtp)

    def val(self,p,m,f):
        cs,cn = self.cost(p,m,f) 
        return f.val(m)*self.r0(p)*(m*p-cs) + (1-f.val(m))*(p-cn)

    def grad(self,p,m,wtp):
        Ks, Kn = self.K
        f = wtp.val(m)
        r = self.r0(p)
        dPdp = 2*self.gamma*r*(Kn*(1-f)**2 +Ks*f**2*self.e_s) + (r-self.gamma*p)*(1-f*(1-m)) +self.gamma*(self.beta[1]*(1-f) +self.beta[0]*f)
        dPdm = r*f*p +r*wtp.grad(m)*(-(1-m)*p +self.beta[1] -self.beta[0] +2*Kn*(1-f)*r -2*Ks*f*self.e_s*r)
        return dPdp, dPdm

    def hess(self,p,m,wtp):
        f = wtp.val(m)
        gf = wtp.grad(m)
        lf = wtp.lap(m)
        r = self.r0(p)
        d2pP = -2*self.gamma**2*(self.K[1]*(1-f)**2 + self.K[0]*self.e_s*f**2) -2*self.gamma*(1- f*(1-m)) 
        dpdmP = (r-self.gamma*p) *(f -(1-m)*gf) + self.gamma*gf*(self.beta[0] -self.beta[1] +4*r*(self.K[0]*f*self.e_s -self.K[1]*(1-f)))
        d2mP = 2*r*gf -2*r**2 *gf**2 *(self.K[1] +self.K[0]*self.e_s) +r*lf*(-(1-m)*p +self.beta[1]-self.beta[0] +2*self.K[1]*(1-f)*r -2*self.K[0]*f*self.e_s*r)
        H = np.array([[d2pP,dpdmP],[dpdmP,d2mP]])
        #print(np.linalg.eig(H)[0])
       
        return H

    def GetConstraint(self,p,m,wtp):
        '''returns concavity condition to ensure that Hessian is negative semidefinite (i.e. that P= objective function is always maximal)
        '''
        H = self.hess(p,m,wtp)
        x = H[0,0] + H[1,1]
        q = H[0,1]**2 -H[0,0]*H[1,1]
        return x, q

    def GradConstraint(self,p,m,wtp):
        '''returns gradient of convexity constraint'''
        H = self.hess(p,m,wtp)
        f = wtp.val(m)
        gf = wtp.grad(m)
        lf = wtp.lap(m)
        Ks,Kn = self.K
        r = self.r0(p)
        grad11 = np.array([0,4*self.gamma**2*gf*(Kn*(1-f)-Ks*self.e_s*f) +2*self.gamma*gf*(1-m)])
        grad12 = np.array([-2*self.gamma*(f-(1-m)*gf) -4*self.gamma**2*(Ks*f*self.e_s -Kn*(1-f))*gf, (r-self.gamma*p)*(2*gf-(1-m)*lf) +self.gamma*lf*(self.beta[0]-self.beta[1] +4*r*(Ks*f*self.e_s -Kn*(1-f))) +4*r* self.gamma*gf**2*(Ks*self.e_s +Kn)]) 
        grad22 = np.array([(r-self.gamma*p)*(2*gf- (1-m)*lf) +
            4*self.gamma*r*((Ks*self.e_s +Kn)*gf**2 
            - (Kn*(1-f)-Ks*f*self.e_s)*lf) 
            -self.gamma*lf*(self.beta[1]-self.beta[0]),
            3*r*p*lf -6*r**2 *(Kn+Ks*self.e_s) *gf*lf +r*wtp.jerk(m) *(-(1-m)*p +self.beta[1] -self.beta[0] +2*(Kn*(1-f) -Ks*f*self.e_s)*r) ]) 
       
        gradq = 2*H[0,1]* grad12 - H[1,1]*grad11 - H[0,0]*grad22
        return gradq

    def cond(self,p,m):
        '''check bounds for p and s'''
        check = (p >= 0) & (p <= self.pmax) & (m >= 0) & (self.r0(p) >= 0) & (E(p,m) >= 0)
        return check

if __name__=="__main__":
    import pandas as pd
    fs = WTP_cubic()
    P = ProfDensity() 
    print(P.GradConstraint(2,1.25,fs))
    #print(P.hess(2,1.25,fs))
