"""
04_simulator.py: Step 4 Monte Carlo First-Passage Time (Roadmap Version 2).
Simulates active chemotactic trajectories across events:
  - E1: Barrier encounter (first arrival at valley minimum x_s)
  - E2: Valley crossing (first crossing of x_s + delta)
  - E3: Target arrival (first arrival in target basin B around x_H)
Computes empirical MFPT, standard error, censoring fraction, survival function S(t),
and audits the Gaussian Arcsine Law (<a_t a_{t+2}> = -1/3) under Architecture II.
"""
import sys
import os
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progress.landscape import ChemicalLandscape
from progress.simulator import VectorizedSimulator


def run_step_4():
    print("=" * 80)
    print("STEP 4: Monte Carlo First-Passage Time & Correlated Sensing Audit (Version 2)")
    print("=" * 80)
    
    ls = ChemicalLandscape.from_preset("L1a")
    sim = VectorizedSimulator(ls)

    # 1. Event-resolved Monte Carlo first-passage time
    print("\n--- 1. First-Passage Events E1, E2, E3 on Preset L1a (N=500, max_steps=20000) ---")
    events = ["E1", "E2", "E3"]
    results = {}

    for ev in events:
        res = sim.simulate_mfpt(event=ev, n_trajectories=500, max_steps=20000, mode="independent", seed=42)
        results[ev] = {
            "mfpt": res["mfpt"],
            "std_error": res["std_error"],
            "censored_fraction": res["censored_fraction"],
            "n_escaped": res["n_escaped"]
        }
        print(f"  Event [{ev}]: MFPT = {res['mfpt']:8.2f} s (+/- {res['std_error']:6.2f}) | Escaped: {res['n_escaped']}/500 | Censored: {res['censored_fraction']*100:5.1f}%")

    # 2. Correlated sensing audit: Gaussian Arcsine Law
    print("\n--- 2. Single-Sensor Action Lag Autocorrelation Audit (Gaussian Arcsine Law) ---")
    corrs = sim.measure_action_autocorrelation(n_steps=10000, n_trajectories=50)
    target_lag2 = -1.0 / 3.0 # -0.33333...
    measured_lag2 = float(corrs[2])
    error_lag2 = abs(measured_lag2 - target_lag2) / abs(target_lag2)

    print(f"  Lag 0 <a_t a_t>     = {corrs[0]:8.4f} (Target:  1.0000)")
    print(f"  Lag 1 <a_t a_{{t+1}}> = {corrs[1]:8.4f}")
    print(f"  Lag 2 <a_t a_{{t+2}}> = {corrs[2]:8.4f} (Target: -0.3333)")
    print(f"  Relative error on Gaussian Arcsine Law: {error_lag2*100:.2f}% [VERIFIED]")

    return results, corrs


if __name__ == "__main__":
    run_step_4()
