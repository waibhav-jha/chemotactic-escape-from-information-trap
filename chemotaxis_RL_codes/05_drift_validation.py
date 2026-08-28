import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from parameters import *

def concentration(x):
    return A1*np.exp(-(x-X1)**2/(2*S1**2))+A2*np.exp(-(x-X2)**2/(2*S2**2))

def dc_dx(x):
    return (-A1*(x-X1)/S1**2*np.exp(-(x-X1)**2/(2*S1**2))
            -A2*(x-X2)/S2**2*np.exp(-(x-X2)**2/(2*S2**2)))

def theory_drift(x):
    g=dc_dx(x)
    return V*np.sign(g)*erf(np.abs(g)*V*DT/(2*SIGMA))

def estimate_drift(x0,n=50000,seed=SEED):
    rng=np.random.default_rng(seed)
    x=np.full(n,x0)
    a=rng.choice([-1,1],n)
    c0=concentration(x)+SIGMA*rng.standard_normal(n)
    c1=concentration(x)+SIGMA*rng.standard_normal(n)
    a=np.where(c1-c0>0,a,-a)
    dx=V*a*DT+np.sqrt(2*D*DT)*rng.standard_normal(n)
    return dx.mean()/DT

if __name__=="__main__":
    xs=np.linspace(X1-1,X2+1,31)
    sim=np.array([estimate_drift(z) for z in xs])
    th=theory_drift(xs)
    plt.figure(figsize=(8,5))
    plt.plot(xs,sim,"o",ms=4,label="simulation")
    plt.plot(xs,th,lw=2,label="analytical benchmark")
    plt.xlabel("x"); plt.ylabel(r"$\mu(x)$")
    plt.legend(); plt.tight_layout(); plt.show()
