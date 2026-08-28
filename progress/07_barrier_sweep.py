"""
07_barrier_sweep.py: Barrier sweep at fixed sigma=0.10 verifying the Arrhenius slope 2/sqrt(pi) = 1.12838.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.markov_chain import ExactMarkovChainSolver
from progress.theory import ContinuumTheory

if __name__ == "__main__":
    print("=== [07_barrier_sweep.py] Barrier Sweep & Arrhenius Slope ===")
    a1_ratios = [0.60, 0.75, 0.90, 1.00, 1.15, 1.30]
    du_sig_list = []
    t_chain_list = []
    t_single_list = []

    print(f"{'a1':>6s} | {'Delta U':>10s} | {'Delta U/sigma':>14s} | {'Exact Chain (s)':>16s} | {'Single Sensor (s)':>18s}")
    print("-" * 75)

    for a1 in a1_ratios:
        p = PhysicalParameters(a1=a1, sigma=0.10, dt=0.01)
        ls = ChemicalLandscape(p)
        chain = ExactMarkovChainSolver(ls, p)
        theory = ContinuumTheory(ls, p)

        res = chain.solve_exact_mfpt()
        t_quad_single = float(theory.mfpt_quadrature(mode="single_sensor"))

        du = float(ls.delta_u_trap)
        du_sig = float(du / p.sigma)

        du_sig_list.append(du_sig)
        t_chain_list.append(float(res["mfpt_point_start"]))
        t_single_list.append(t_quad_single)

        print(f"{a1:6.2f} | {du:10.4f} | {du_sig:14.2f} | {res['mfpt_point_start']:16.2f} | {t_quad_single:18.2f}")

    x_data = np.array(du_sig_list, dtype=np.float64)
    y_chain = np.log(np.array(t_chain_list, dtype=np.float64))
    y_single = np.log(np.array(t_single_list, dtype=np.float64))

    fit_indep = np.polyfit(x_data, y_chain, 1)
    fit_single = np.polyfit(x_data, y_single, 1)

    print("\n--- Fitted Scaling Exponents ---")
    print(f"Independent Readings Slope: {fit_indep[0]:.4f} (Target: 2/sqrt(pi) = 1.1284)")
    print(f"Single Sensor Audit Slope:  {fit_single[0]:.4f} (Target: ~1.554)")
