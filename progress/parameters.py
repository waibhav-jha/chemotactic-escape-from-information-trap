"""
Physical parameters, domain specifications, landscape presets, and discretization settings
for Chemotactic Escape from Information Traps (Steps 1 to 8).
All modules share this single source of truth to guarantee matched conventions.
"""
from dataclasses import dataclass
import sys
import os

# Add directory containing progress to sys.path if not present
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class PhysicalParameters:
    # Landscape presets: (x1, x2, a1, a2, s1, s2)
    x1: float = -1.5
    x2: float = 1.5
    a1: float = 1.0
    a2: float = 1.0
    s1: float = 0.7
    s2: float = 0.7

    # Motility and environment parameters
    v: float = 1.0          # Active swimming speed
    D: float = 0.0          # Environmental Brownian diffusivity (default D=0 for exact lattice match)
    dt: float = 0.01        # Timestep
    sigma: float = 0.10     # Sensing noise standard deviation

    # Spatial domain and boundary settings
    x_left: float = -5.0    # Reflecting boundary
    x_right: float = 5.0    # Outer right domain bound
    absorbing_delta: float = 0.0  # Absorbing boundary offset: x_b = x_min + delta

    # Exact grid matching: h = v * dt / K
    K_grid: int = 1         # Grid matching integer: agent moves exactly K bins per step

    # Monte Carlo simulation settings
    N_mc: int = 10000       # Trajectory count
    max_steps: int = 500000 # Max step safety cap per trajectory
    seed: int = 42

    @property
    def h(self) -> float:
        """Grid spacing matching the deterministic step displacement exactly."""
        return (self.v * self.dt) / float(self.K_grid)


# Standard Landscape Presets from Analytical & Computational foundations
LANDSCAPE_PRESETS = {
    "shallow": {
        "x1": -1.5, "x2": 1.5, "a1": 0.70, "a2": 1.30, "s1": 0.85, "s2": 0.70,
        "name": "Shallow Trap (Delta U ~ 0.15 - 0.20)"
    },
    "medium": {
        "x1": -1.5, "x2": 1.5, "a1": 1.00, "a2": 1.00, "s1": 0.70, "s2": 0.70,
        "name": "Medium Trap (Delta U ~ 0.5443)"
    },
    "deep": {
        "x1": -1.5, "x2": 1.5, "a1": 1.40, "a2": 1.00, "s1": 0.65, "s2": 0.70,
        "name": "Deep Trap (Delta U ~ 0.85 - 1.10)"
    }
}


# Phase 2 (Step 9+) Multi-Bit & Information-Theoretic Sweeps
MEMORY_BIT_SWEEP = [0, 1, 2, 3, 4, 5, 6, 8, 10]

DUAL_NOISE_SWEEP = {
    "sigma_vals": [0.02, 0.05, 0.10, 0.20, 0.40],
    "D_vals": [0.0, 0.001, 0.01, 0.05, 0.10]
}

ASYMMETRY_RATIOS = [0.5, 0.75, 1.0, 1.33, 2.0]

THERMODYNAMIC_PARAMS = {
    "kB_T": 1.0,               # Normalized thermal energy unit
    "ln2": 0.69314718056,      # Natural log 2 (Landauer constant per bit)
    "beta_sweep": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
}
