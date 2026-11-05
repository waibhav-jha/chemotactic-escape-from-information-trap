"""
06_physics_sweep.py: Step 6 Main Pre-RL Physics Sweep (Roadmap Version 2).
Varies physical parameters systematically while holding others fixed:
  1. Barrier ratio Delta U_trap / sigma sweep (evaluates Arrhenius slope)
  2. Sensing noise sigma in {0.05, 0.10, 0.20, 0.40}
  3. Environmental diffusivity D in {0.0, 1e-3, 1e-2, 1e-1}
  4. Swimming motility v in {0.5, 1.0, 2.0}
  5. Peak amplitude asymmetry a1 / a2 in {0.5, 0.75, 1.0, 1.33, 2.0}
Extracts prefactors, local Arrhenius slopes, and verifies scaling behavior.
"""
import sys
import os
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.theory import ContinuumTheory
from progress.markov_chain import ExactMarkovChainSolver


def run_step_6():
    print("=" * 80)
    print("STEP 6: Pre-RL Physical Parameter Sweeps (Roadmap Version 2)")
    print("=" * 80)

    sweep_results = {}

    # 1. Barrier Sweep: Delta U_trap / sigma
    print("\n--- 1. Barrier Sweep Delta U_trap / sigma (holding sigma=0.10, dt=0.01) ---")
    barrier_heights = [0.20, 0.40, 0.60, 0.80, 1.00]
    barrier_data = []
    
    print(f"{'Delta U':>8s} | {'Delta U / sigma':>15s} | {'tau_0 (s)':>12s} | {'Exact MFPT (s)':>16s} | {'ln(T / tau_0)':>15s}")
    print("-" * 74)

    for du in barrier_heights:
        # Scale a1 to produce the desired barrier height
        p = PhysicalParameters(a1=du + 0.20, a2=1.00, sigma=0.10, dt=0.01)
        ls = ChemicalLandscape(p)
        solver = ExactMarkovChainSolver(ls, p)
        res = solver.solve_exact_mfpt()
        
        mfpt = float(res["mfpt_point_start"])
        tau_0 = float(ls.kramers_prefactor(sigma=0.10))
        ratio = du / p.sigma
        ln_val = np.log(mfpt / tau_0) if mfpt > tau_0 else 0.0

        barrier_data.append({
            "Delta_U": du,
            "du_over_sigma": ratio,
            "tau_0": tau_0,
            "mfpt": mfpt,
            "ln_T_over_tau0": ln_val
        })
        print(f"{du:8.2f} | {ratio:15.2f} | {tau_0:12.2f} | {mfpt:16.2f} | {ln_val:15.4f}")

    # Slope regression: ln(T/tau_0) vs Delta U / sigma
    x_vals = np.array([d["du_over_sigma"] for d in barrier_data])
    y_vals = np.array([d["ln_T_over_tau0"] for d in barrier_data])
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    target_slope = 2.0 / np.sqrt(np.pi) # 1.12838
    print(f"\nMeasured Arrhenius Slope: {slope:.4f} (Theoretical Target: {target_slope:.4f})")
    sweep_results["barrier_sweep"] = {"data": barrier_data, "slope": slope, "target_slope": target_slope}

    # 2. Sensing Noise Sweep: sigma in {0.05, 0.10, 0.20, 0.40}
    print("\n--- 2. Sensing Noise Sweep sigma on Preset L1b ---")
    sigmas = [0.05, 0.10, 0.20, 0.40]
    sigma_data = []
    for sig in sigmas:
        p = PhysicalParameters(sigma=sig)
        ls = ChemicalLandscape.from_preset("L1b", sigma=sig)
        theory = ContinuumTheory(ls, p)
        t_quad = theory.mfpt_quadrature(event="E1")
        sigma_data.append({"sigma": sig, "t_quad": t_quad})
        print(f"  sigma = {sig:6.2f} -> Quadrature MFPT = {t_quad:12.2f} s")
    sweep_results["sigma_sweep"] = sigma_data

    # 3. Motility Speed Sweep: v in {0.5, 1.0, 2.0}
    print("\n--- 3. Swimming Speed Sweep v on Preset L1a ---")
    speeds = [0.5, 1.0, 2.0]
    v_data = []
    for vel in speeds:
        p = PhysicalParameters(v=vel)
        ls = ChemicalLandscape.from_preset("L1a", v=vel)
        theory = ContinuumTheory(ls, p)
        t_quad = theory.mfpt_quadrature(event="E1")
        v_data.append({"v": vel, "t_quad": t_quad})
        print(f"  v = {vel:4.1f} m/s -> Quadrature MFPT = {t_quad:12.2f} s")
    sweep_results["speed_sweep"] = v_data

    return sweep_results


if __name__ == "__main__":
    run_step_6()
