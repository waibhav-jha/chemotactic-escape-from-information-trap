import numpy as np
from parameters import *

def concentration(x):
    return (A1*np.exp(-(x-X1)**2/(2*S1**2))
            + A2*np.exp(-(x-X2)**2/(2*S2**2)))

def simulate_1bit(Ntraj=NTRAJ, nsteps=5000, x0=X1,
                  v=V, D_=D, dt=DT, sigma=SIGMA, seed=SEED):
    rng=np.random.default_rng(seed)
    x=np.full(Ntraj,x0,dtype=float)
    action=rng.choice([-1,1],size=Ntraj)
    c_prev=concentration(x)+sigma*rng.standard_normal(Ntraj)
    traj=np.empty((nsteps+1,Ntraj)); traj[0]=x

    for t in range(nsteps):
        c_now=concentration(x)+sigma*rng.standard_normal(Ntraj)
        action=np.where(c_now-c_prev>0,action,-action)
        x += v*action*dt + np.sqrt(2*D_*dt)*rng.standard_normal(Ntraj)
        traj[t+1]=x
        c_prev=c_now
    return traj

def simulate_0bit(Ntraj=NTRAJ, nsteps=5000, x0=X1,
                  v=V, D_=D, dt=DT, sigma=SIGMA,
                  c_ref=0.5, seed=SEED):
    rng=np.random.default_rng(seed)
    x=np.full(Ntraj,x0,dtype=float)
    traj=np.empty((nsteps+1,Ntraj)); traj[0]=x

    for t in range(nsteps):
        c_now=concentration(x)+sigma*rng.standard_normal(Ntraj)
        action=np.where(c_now>c_ref,1,-1)
        x += v*action*dt + np.sqrt(2*D_*dt)*rng.standard_normal(Ntraj)
        traj[t+1]=x
    return traj
