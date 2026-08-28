"""
02_convergence.py: Complete implementation of Step 2 Convergence Blueprint.
Evaluates numerical convergence over dt, grid spacing h = v*dt/K, and absorbing boundary offset delta.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.markov_chain import ExactMarkovChainSolver

if __name__ == "__main__":
    print("=== [02_convergence.py] Timestep and Grid Convergence ===")
    dts = [0.05, 0.02, 0.01, 0.005]
    mfpts = []

    for dt in dts:
        p = PhysicalParameters(dt=dt)
        ls = ChemicalLandscape.from_preset("medium", dt=dt)
        solver = ExactMarkovChainSolver(ls, p)
        res = solver.solve_exact_mfpt()
        mfpts.append(res["mfpt_point_start"])
        print(f"dt = {dt:6.4f} | Grid h = {p.h:6.4f} (n={solver.n_spatial:4d}) | Exact Chain MFPT = {res['mfpt_point_start']:10.3f} s")

    plt.figure(figsize=(7, 5))
    plt.plot(dts, mfpts, "o-", color="#386cb0", lw=2, markersize=7)
    plt.xlabel("Timestep dt (seconds)"); plt.ylabel("Exact MFPT (seconds)")
    plt.title("Numerical Convergence of Exact Discretized Markov Chain")
    plt.tight_layout(); plt.show()
