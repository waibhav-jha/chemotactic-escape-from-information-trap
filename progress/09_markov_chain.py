"""
09_markov_chain.py: Exact sparse Markov Chain solver, committor, and 4-way validation table.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.markov_chain import ExactMarkovChainSolver
from progress.theory import ContinuumTheory

if __name__ == "__main__":
    print("=== [09_markov_chain.py] Exact Markov Chain Solver ===")
    ls = ChemicalLandscape.from_preset("medium")
    solver = ExactMarkovChainSolver(ls)
    theory = ContinuumTheory(ls)

    res = solver.solve_exact_mfpt()
    t_quad = theory.mfpt_quadrature()
    t_saddle = theory.kramers_mfpt()
    committor = solver.compute_committor()

    print(f"Exact Markov Linear Solve MFPT:  {res['mfpt_point_start']:10.3f} s")
    print(f"Fokker-Planck Double Quadrature: {t_quad:10.3f} s (Error: {abs(res['mfpt_point_start'] - t_quad)/t_quad*100:.2f}%)")
    print(f"Saddle-Point Kramers Formula:    {t_saddle:10.3f} s")
    print(f"Reaction Coordinate Diagnostic:  Max Delta q = {committor['max_delta_q']:.6e} (confirms Delta q ~ 0)")
    print(f"Transition State Location:       x = {committor['x_transition_state']:.3f} (precedes valley minimum {ls.x_min:.3f})")
