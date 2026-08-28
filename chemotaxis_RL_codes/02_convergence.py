import numpy as np
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
from parameters import *

sim = SourceFileLoader("sim","04_simulator.py").load_module()

def mfpt_from_trajectories(traj, escape_x=ESCAPE_X, dt=DT):
    hit = traj >= escape_x
    exists = hit.any(axis=0)
    first = np.argmax(hit, axis=0)
    times = first[exists]*dt
    return np.mean(times) if len(times) else np.nan

if __name__ == "__main__":
    dts = [0.05,0.02,0.01,0.005]
    values = []
    for dt in dts:
        traj = sim.simulate_1bit(Ntraj=3000,nsteps=10000,
                                 dt=dt,seed=SEED)
        values.append(mfpt_from_trajectories(traj,dt=dt))
        print(f"dt={dt:g}, MFPT={values[-1]}")
    plt.figure(figsize=(7,5))
    plt.plot(dts,values,"o-")
    plt.xscale("log"); plt.xlabel("dt"); plt.ylabel("MFPT")
    plt.title("Timestep convergence"); plt.tight_layout(); plt.show()
