"""
Physical parameters, domain specifications, landscape presets, and discretization settings
for Chemotactic Escape from Information Traps (Roadmap Version 2: Steps 1 to 16).
All modules share this single source of truth to guarantee matched physical conventions.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
import sys
import os

# Add directory containing progress to sys.path if not present
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class PhysicalParameters:
    # Landscape parameters: (x1, x2, a1, a2, s1, s2)
    x1: float = -1.5
    x2: float = 1.5
    a1: float = 1.0
    a2: float = 1.0
    s1: float = 0.70
    s2: float = 0.70

    # Motility and environment parameters
    v: float = 1.0          # Active swimming speed
    D: float = 0.0          # Environmental Brownian diffusivity (D=0 for exact lattice match)
    dt: float = 0.01        # Movement timestep
    dt_s: float = 0.01      # Sensor measurement interval (distinct from movement dt)
    sigma: float = 0.10     # Sensing noise standard deviation

    # Spatial domain and boundary settings
    x_left: float = -5.0    # Reflecting boundary r
    x_right: float = 5.0    # Outer right domain bound
    absorbing_delta: float = 0.0  # Absorbing boundary offset for crossing event E2: x_s + delta
    target_x: float = 1.5   # Global target peak x_H center
    target_radius: float = 0.20 # Target basin radius B = [target_x - r, target_x + r]

    # Exact grid matching: h = v * dt / K
    K_grid: int = 1         # Grid matching integer: agent moves exactly K bins per step

    # Monte Carlo simulation settings
    N_mc: int = 10000       # Trajectory count
    max_steps: int = 500000 # Max step safety cap per trajectory
    seed: int = 42

    # Stochastic calculus convention parameter alpha: 1.0 = Ito, 0.5 = Stratonovich, 0.0 = Hänggi-Klimontovich
    alpha_conv: float = 1.0

    @property
    def h(self) -> float:
        """Grid spacing matching the deterministic step displacement exactly."""
        return (self.v * self.dt) / float(self.K_grid)


# Standard Landscape Presets from Roadmap Version 2 (L1a, L1b, L1c)
LANDSCAPE_PRESETS: Dict[str, Dict[str, Any]] = {
    # L1a: Shallow trap
    "L1a": {
        "x1": -1.5, "x2": 1.5, "a1": 0.70, "a2": 1.30, "s1": 0.85, "s2": 0.70,
        "name": "L1a: Shallow Trap (Delta U ~ 0.425)"
    },
    "shallow": {
        "x1": -1.5, "x2": 1.5, "a1": 0.70, "a2": 1.30, "s1": 0.85, "s2": 0.70,
        "name": "L1a: Shallow Trap (Delta U ~ 0.425)"
    },
    # L1b: Medium trap
    "L1b": {
        "x1": -1.5, "x2": 1.5, "a1": 1.00, "a2": 1.00, "s1": 0.70, "s2": 0.70,
        "name": "L1b: Medium Trap (Delta U ~ 0.799)"
    },
    "medium": {
        "x1": -1.5, "x2": 1.5, "a1": 1.00, "a2": 1.00, "s1": 0.70, "s2": 0.70,
        "name": "L1b: Medium Trap (Delta U ~ 0.799)"
    },
    # L1c: Deep trap
    "L1c": {
        "x1": -1.5, "x2": 1.5, "a1": 1.00, "a2": 1.40, "s1": 0.65, "s2": 0.70,
        "name": "L1c: Deep Trap (Delta U ~ 0.802, High Barrier)"
    },
    "deep": {
        "x1": -1.5, "x2": 1.5, "a1": 1.00, "a2": 1.40, "s1": 0.65, "s2": 0.70,
        "name": "L1c: Deep Trap (Delta U ~ 0.802, High Barrier)"
    },
    # L2: Mirrored orientation (reverses gradient direction to test transfer without target cheat)
    "L2": {
        "x1": 1.5, "x2": -1.5, "a1": 1.00, "a2": 1.00, "s1": 0.70, "s2": 0.70,
        "name": "L2: Mirrored Orientation (Reversed Coordinates)"
    },
    # L3: Rugged / multipeak landscape with intermediate deceptive local sub-well
    "L3": {
        "x1": -1.5, "x2": 1.5, "a1": 1.00, "a2": 1.20, "s1": 0.70, "s2": 0.70,
        "x3": 0.20, "a3": 0.35, "s3": 0.25, # Deceptive local sub-peak in valley
        "name": "L3: Rugged Multipeak Landscape"
    }
}


# Events defined in Version 2 Roadmap
EVENT_DEFINITIONS = {
    "E1": "Barrier Encounter: First arrival at valley minimum x_s",
    "E2": "Valley Crossing: First crossing of x_s + delta (delta > 0)",
    "E3": "Successful Escape: First arrival in target basin B around x_H"
}


# Sweeps across physics & memory dimensions
MEMORY_BIT_SWEEP: List[int] = [0, 1, 2, 3, 4, 5, 6, 8, 10]

DUAL_NOISE_SWEEP = {
    "sigma_vals": [0.02, 0.05, 0.10, 0.20, 0.40],
    "D_vals": [0.0, 0.001, 0.01, 0.05, 0.10]
}

ASYMMETRY_RATIOS = [0.50, 0.75, 1.00, 1.33, 2.00]

TIMESTEP_SWEEP = [0.10, 0.05, 0.02, 0.01, 0.005]
SPATIAL_GRID_SWEEP = [50, 100, 200, 400]
NMC_SWEEP = [1000, 10000, 100000]

THERMODYNAMIC_PARAMS = {
    "kB_T": 1.0,               # Normalized thermal energy unit
    "ln2": 0.69314718056,      # Natural log 2 (Landauer constant per bit)
    "beta_sweep": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
}
