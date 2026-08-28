import numpy as np
from parameters import *

def concentration(x):
    return A1*np.exp(-(x-X1)**2/(2*S1**2))+A2*np.exp(-(x-X2)**2/(2*S2**2))

def mfpt_1bit(Ntraj=NTRAJ,x0=X1,escape_x=ESCAPE_X,
              v=V,D_=D,dt=DT,sigma=SIGMA,
              max_steps=MAX_STEPS,seed=SEED):
    rng=np.random.default_rng(seed)
    x=np.full(Ntraj,x0,dtype=float)
    action=rng.choice([-1,1],Ntraj)
    c_prev=concentration(x)+sigma*rng.standard_normal(Ntraj)
    escaped=np.zeros(Ntraj,bool)
    times=np.full(Ntraj,np.nan)

    for step in range(max_steps):
        active=~escaped
        if not active.any(): break
        idx=np.where(active)[0]
        c_now=concentration(x[idx])+sigma*rng.standard_normal(len(idx))
        action[idx]=np.where(c_now-c_prev[idx]>0,action[idx],-action[idx])
        x[idx]+=v*action[idx]*dt+np.sqrt(2*D_*dt)*rng.standard_normal(len(idx))
        new=x[idx]>=escape_x
        new_idx=idx[new]
        times[new_idx]=(step+1)*dt
        escaped[new_idx]=True
        c_prev[idx]=c_now

    valid=times[np.isfinite(times)]
    return (np.mean(valid) if len(valid) else np.nan), valid

if __name__=="__main__":
    mfpt,times=mfpt_1bit()
    print("MFPT =",mfpt)
    print("escaped =",len(times),"/",NTRAJ)
