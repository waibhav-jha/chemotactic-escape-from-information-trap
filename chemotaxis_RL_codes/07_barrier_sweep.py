import numpy as np
from scipy.optimize import minimize_scalar
from parameters import *

def concentration(x,a1,a2):
    return a1*np.exp(-(x-X1)**2/(2*S1**2))+a2*np.exp(-(x-X2)**2/(2*S2**2))

def barrier(a1,a2):
    r=minimize_scalar(lambda z: concentration(z,a1,a2),
                      bounds=(X1,X2),method="bounded")
    return concentration(X1,a1,a2)-concentration(r.x,a1,a2)

if __name__=="__main__":
    ratios=[.5,.75,1,1.33,2]
    print("a1/a2      DeltaU")
    for r in ratios:
        print(f"{r:5.2f}      {barrier(r,1):.6f}")
    print("\nUse the MFPT evaluator for each landscape and fit:")
    print("log(MFPT) = intercept + slope*(DeltaU/SIGMA)")
    print("Benchmark slope: 2/sqrt(pi) = 1.128...")
