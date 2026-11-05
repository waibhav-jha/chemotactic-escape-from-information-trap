"""
01_landscape.py: Step 1 Blueprint (Roadmap Version 2).
Constructs double-Gaussian chemical landscapes, analytical derivatives c'(x) and c''(x),
finds actual extrema (x_L, x_s, x_H), computes Delta U_trap and Delta c_target,
and verifies c'(x) = 0 at extrema across presets L1a, L1b, L1c, L2, and L3.
"""
import sys
import os
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progress.landscape import ChemicalLandscape


def run_step_1():
    print("=" * 80)
    print("STEP 1: Construct Chemical Landscapes & Extrema Geometry (Roadmap Version 2)")
    print("=" * 80)
    
    presets = ["L1a", "L1b", "L1c", "L2", "L3"]
    results = {}
    
    for name in presets:
        ls = ChemicalLandscape.from_preset(name)
        s = ls.summary()
        results[name] = s
        print(f"\n--- Preset: {name} ({s.get('preset')}) ---")
        print(f"  Lower Peak x_L     = {s['x_L (lower peak)']:10.4f} | c(x_L) = {s['c(x_L)']:10.4f} | c''(x_L) = {s['c''(x_L)']:10.4f}")
        print(f"  Valley Saddle x_s  = {s['x_s (valley)']:10.4f} | c(x_s) = {s['c(x_s)']:10.4f} | c''(x_s) = {s['c''(x_s)']:10.4f}")
        print(f"  Higher Peak x_H    = {s['x_H (higher peak)']:10.4f} | c(x_H) = {s['c(x_H)']:10.4f} | c''(x_H) = {s['c''(x_H)']:10.4f}")
        print(f"  Barrier Delta U_trap   = {s['Delta_U_trap']:10.4f}")
        print(f"  Target Delta c_target  = {s['Delta_c_target']:10.4f}")
        print(f"  Kramers tau_0 (s)      = {s['tau_0 (sigma=0.10)']:10.4f}")
        
        # Verify derivative vanishes at extrema
        grad_L = float(ls.dc(ls.x_L))
        grad_s = float(ls.dc(ls.x_s))
        grad_H = float(ls.dc(ls.x_H))
        assert abs(grad_L) < 1e-4, f"c'(x_L) != 0: {grad_L}"
        assert abs(grad_s) < 1e-4, f"c'(x_s) != 0: {grad_s}"
        assert abs(grad_H) < 1e-4, f"c'(x_H) != 0: {grad_H}"
        print(f"  Extrema Verification: c'(x_L)={grad_L:.2e}, c'(x_s)={grad_s:.2e}, c'(x_H)={grad_H:.2e} [VERIFIED]")

    return results


if __name__ == "__main__":
    run_step_1()
