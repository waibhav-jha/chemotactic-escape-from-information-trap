"""
06_mfpt.py: Computes empirical first-passage time distributions and checks against exact Markov chain.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.simulator import VectorizedSimulator
from progress.markov_chain import ExactMarkovChainSolver

if __name__ == "__main__":
    print("=== [06_mfpt.py] Mean First-Passage Time Benchmark ===")
    p = PhysicalParameters(sigma=0.35, dt=0.01)
    ls = ChemicalLandscape.from_preset("medium", sigma=0.35, dt=0.01)
    sim = VectorizedSimulator(ls, p)
    chain = ExactMarkovChainSolver(ls, p)

    print("Running Monte Carlo simulation (10,000 trajectories)...")
    mc_res = sim.simulate_mfpt(n_trajectories=10000, mode="independent")
    chain_res = chain.solve_exact_mfpt()

    print(f"Monte Carlo MFPT:       {mc_res['mfpt']:10.3f} s (+/- {mc_res['std_error']:.3f})")
    print(f"Exact Markov Chain MFPT:{chain_res['mfpt_point_start']:10.3f} s")
    print(f"QSD Escape Time:        {chain_res['mfpt_qsd']:10.3f} s")
    print(f"Slow Mode tau_esc:      {chain_res['tau_esc_spectral']:10.3f} s")
    print(f"Fast Mode tau_rel:      {chain_res['tau_rel']:10.3f} s")
    print(f"Spectral Separation:    {chain_res['timescale_separation']:.4e}")
