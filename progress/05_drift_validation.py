"""
05_drift_validation.py: Validates simulated 1-bit drift against analytical theory
mu(x) = v * sign(c') * erf(|c'|*v*dt / (2*sigma)) and executes the Correlated Sensing Audit.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.landscape import ChemicalLandscape
from progress.theory import ContinuumTheory
from progress.simulator import VectorizedSimulator

if __name__ == "__main__":
    print("=== [05_drift_validation.py] Drift Profile & Correlated Sensing Audit ===")
    ls = ChemicalLandscape.from_preset("medium")
    theory = ContinuumTheory(ls)
    sim = VectorizedSimulator(ls)

    xs = np.linspace(-3.5, 3.5, 31)
    sim_data_indep = sim.measure_drift_and_diffusivity(xs, n_samples_per_point=30000, mode="independent")
    sim_data_single = sim.measure_drift_and_diffusivity(xs, n_samples_per_point=30000, mode="single_sensor")

    th_indep = theory.b1_drift(xs, mode="independent")
    th_single = theory.b1_drift(xs, mode="single_sensor")

    plt.figure(figsize=(9, 5))
    plt.plot(xs, sim_data_indep["mu_measured"], "o", label="Monte Carlo (Independent Readings)", color="#1b9e77")
    plt.plot(xs, th_indep, "-", label="Theory: mu = v*erf(...) (Independent)", color="#1b9e77", lw=2)
    plt.plot(xs, sim_data_single["mu_measured"], "s", label="Monte Carlo (Single Sensor)", color="#d95f02")
    plt.plot(xs, th_single, "--", label="Theory: kappa_mu * mu (kappa_mu = 0.7208)", color="#d95f02", lw=2)
    plt.axhline(0, color="k", ls=":", alpha=0.5)
    plt.xlabel("Position x"); plt.ylabel(r"Marginal Drift $\mu(x)$")
    plt.title("Drift Validation: Independent Readings vs Correlated Single Sensor")
    plt.legend(); plt.tight_layout(); plt.show()
