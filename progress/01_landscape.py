"""
01_landscape.py: Complete implementation of Step 1 Landscape Blueprint.
Plots concentration c(x), gradient c'(x), curvature c''(x), finds extrema,
and prints Kramers prefactor tau_0 across Shallow, Medium, and Deep presets.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.landscape import ChemicalLandscape

if __name__ == "__main__":
    print("=== [01_landscape.py] Chemical Landscape & Barrier Extrema ===")
    for preset_name in ["shallow", "medium", "deep"]:
        ls = ChemicalLandscape.from_preset(preset_name)
        s = ls.summary()
        print(f"\n--- Preset: {preset_name.upper()} ---")
        print(f"  x1 (peak 1)  = {s['x1_peak']:10.4f} | c(x1)  = {s['c(x1)']:10.4f} | c''(x1)  = {s['c''(x1)']:10.4f}")
        print(f"  x_min (trap) = {s['x_min']:10.4f} | c(xmin)= {s['c(x_min)']:10.4f} | c''(xmin)= {s['c''(x_min)']:10.4f}")
        print(f"  x2 (peak 2)  = {s['x2_peak']:10.4f} | c(x2)  = {s['c(x2)']:10.4f}")
        print(f"  Barrier Delta U_trap = {s['Delta_U_trap']:10.4f}")
        print(f"  Kramers Prefactor tau_0 (sigma=0.10) = {s['tau_0 (sigma=0.10)']:10.4f} s")

    # Plot
    ls_med = ChemicalLandscape.from_preset("medium")
    x = np.linspace(-4.5, 4.5, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x, ls_med.c(x), lw=2.5, color="#1b9e77", label="c(x)")
    plt.scatter([ls_med.x1_peak, ls_med.x_min, ls_med.x2_peak],
                [ls_med.c_x1, ls_med.c_xmin, ls_med.c_x2],
                color="#d95f02", s=70, zorder=5, label="Extrema (x1, x_min, x2)")
    plt.axvline(ls_med.x_min, color="k", ls="--", alpha=0.6, label=f"Valley Minimum x_min={ls_med.x_min:.2f}")
    plt.title("Double-Gaussian Chemical Landscape (Medium Preset)")
    plt.xlabel("Position x"); plt.ylabel("Concentration c(x)")
    plt.legend(); plt.tight_layout(); plt.show()
