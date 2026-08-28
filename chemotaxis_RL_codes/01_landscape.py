import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from parameters import *

def concentration(x, a1=A1, a2=A2):
    return (a1*np.exp(-(x-X1)**2/(2*S1**2))
            + a2*np.exp(-(x-X2)**2/(2*S2**2)))

def dc_dx(x, a1=A1, a2=A2):
    return (-a1*(x-X1)/S1**2*np.exp(-(x-X1)**2/(2*S1**2))
            -a2*(x-X2)/S2**2*np.exp(-(x-X2)**2/(2*S2**2)))

def find_barrier(a1=A1, a2=A2):
    r = minimize_scalar(lambda z: concentration(z,a1,a2),
                        bounds=(X1,X2), method="bounded")
    return r.x

if __name__ == "__main__":
    x = np.linspace(XMIN_DOMAIN, XMAX_DOMAIN, NX)
    c = concentration(x)
    xm = find_barrier()
    du = concentration(X1)-concentration(xm)

    print(f"x1 = {X1:.6f}")
    print(f"x_min = {xm:.6f}")
    print(f"x2 = {X2:.6f}")
    print(f"Delta U_trap = {du:.6f}")

    plt.figure(figsize=(8,5))
    plt.plot(x,c,lw=2)
    plt.scatter([X1,xm,X2],
                [concentration(X1),concentration(xm),concentration(X2)],
                s=60)
    plt.axvline(xm,ls="--",alpha=.7)
    plt.xlabel("x"); plt.ylabel("c(x)")
    plt.title("Double-Gaussian chemical landscape")
    plt.tight_layout(); plt.show()
