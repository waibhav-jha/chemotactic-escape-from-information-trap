"""
Step 1: Chemical Landscape & Barrier Geometry (Roadmap Version 2).
Constructs c(x), exact analytical derivatives c'(x) and c''(x), finds actual extrema:
  - Lower peak x_L (local attractor / trap)
  - Valley saddle x_s (deceptive intermediate barrier)
  - Higher peak x_H (global nutrient target)
Computes trap barrier Delta U_trap = c(x_L) - c(x_s), target difference Delta c_target = c(x_H) - c(x_L),
and evaluates Kramers curvatures across presets L1a, L1b, L1c, L2 (mirrored), and L3 (rugged).
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize_scalar
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progress.parameters import PhysicalParameters, LANDSCAPE_PRESETS


class ChemicalLandscape:
    def __init__(self, params: Optional[PhysicalParameters] = None, preset_name: Optional[str] = None):
        if params is None:
            params = PhysicalParameters()
        self.params = params
        self.preset_name = preset_name or "custom"
        self.x1_center: float = params.x1
        self.x2_center: float = params.x2
        self.a1: float = params.a1
        self.a2: float = params.a2
        self.s1: float = params.s1
        self.s2: float = params.s2

        # Optional 3rd Gaussian for rugged L3 landscapes
        self.x3_center: float = getattr(params, "x3", 0.0)
        self.a3: float = getattr(params, "a3", 0.0)
        self.s3: float = getattr(params, "s3", 0.25)

        # Find actual numerical extrema
        self.x_L, self.x_s, self.x_H = self._find_extrema()
        
        # Aliases for backward compatibility
        self.x1_peak = self.x_L
        self.x_min = self.x_s
        self.x2_peak = self.x_H

        self.c_xL: float = float(self.c(self.x_L))
        self.c_xs: float = float(self.c(self.x_s))
        self.c_xH: float = float(self.c(self.x_H))
        
        # Aliases
        self.c_x1 = self.c_xL
        self.c_xmin = self.c_xs
        self.c_x2 = self.c_xH

        self.delta_u_trap: float = self.c_xL - self.c_xs
        self.delta_c_target: float = self.c_xH - self.c_xL

        # Curvatures at extrema
        self.curv_xL: float = float(self.d2c(self.x_L))     # Negative for peak
        self.curv_xs: float = float(self.d2c(self.x_s))     # Positive for minimum
        self.curv_xH: float = float(self.d2c(self.x_H))     # Negative for higher peak

        # Aliases
        self.curv_x1 = self.curv_xL
        self.curv_xmin = self.curv_xs
        self.curv_x2 = self.curv_xH

    @classmethod
    def from_preset(cls, preset_name: str, **override_params) -> "ChemicalLandscape":
        preset = LANDSCAPE_PRESETS[preset_name]
        p_dict = {
            "x1": preset["x1"],
            "x2": preset["x2"],
            "a1": preset["a1"],
            "a2": preset["a2"],
            "s1": preset["s1"],
            "s2": preset["s2"],
        }
        p_dict.update(override_params)
        p = PhysicalParameters(**p_dict)
        if "x3" in preset:
            setattr(p, "x3", preset["x3"])
            setattr(p, "a3", preset["a3"])
            setattr(p, "s3", preset["s3"])
        return cls(p, preset_name=preset_name)

    def c(self, x: Any) -> Any:
        """Chemical concentration field c(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        g1 = self.a1 * np.exp(- (x_arr - self.x1_center)**2 / (2.0 * self.s1**2))
        g2 = self.a2 * np.exp(- (x_arr - self.x2_center)**2 / (2.0 * self.s2**2))
        res = g1 + g2
        if self.a3 > 0.0:
            res += self.a3 * np.exp(- (x_arr - self.x3_center)**2 / (2.0 * self.s3**2))
        return res

    def dc(self, x: Any) -> Any:
        """First spatial derivative (gradient) c'(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        t1 = - self.a1 * (x_arr - self.x1_center) / (self.s1**2) * np.exp(- (x_arr - self.x1_center)**2 / (2.0 * self.s1**2))
        t2 = - self.a2 * (x_arr - self.x2_center) / (self.s2**2) * np.exp(- (x_arr - self.x2_center)**2 / (2.0 * self.s2**2))
        res = t1 + t2
        if self.a3 > 0.0:
            res += - self.a3 * (x_arr - self.x3_center) / (self.s3**2) * np.exp(- (x_arr - self.x3_center)**2 / (2.0 * self.s3**2))
        return res

    def d2c(self, x: Any) -> Any:
        """Second spatial derivative (curvature) c''(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        g1 = np.exp(- (x_arr - self.x1_center)**2 / (2.0 * self.s1**2))
        g2 = np.exp(- (x_arr - self.x2_center)**2 / (2.0 * self.s2**2))
        t1 = (self.a1 / self.s1**2) * (((x_arr - self.x1_center)**2 / self.s1**2) - 1.0) * g1
        t2 = (self.a2 / self.s2**2) * (((x_arr - self.x2_center)**2 / self.s2**2) - 1.0) * g2
        res = t1 + t2
        if self.a3 > 0.0:
            g3 = np.exp(- (x_arr - self.x3_center)**2 / (2.0 * self.s3**2))
            res += (self.a3 / self.s3**2) * (((x_arr - self.x3_center)**2 / self.s3**2) - 1.0) * g3
        return res

    def _find_extrema(self) -> Tuple[float, float, float]:
        """Locates lower peak x_L, valley saddle x_s, and higher peak x_H."""
        x_left_bound = min(self.x1_center, self.x2_center)
        x_right_bound = max(self.x1_center, self.x2_center)

        # Valley minimum x_s between centers
        res_min = minimize_scalar(lambda z: float(self.c(z)), bounds=(x_left_bound, x_right_bound), method="bounded")
        x_s = float(res_min.x)

        # Left peak
        res_p1 = minimize_scalar(lambda z: -float(self.c(z)), bounds=(self.params.x_left, x_s), method="bounded")
        p1 = float(res_p1.x)

        # Right peak
        res_p2 = minimize_scalar(lambda z: -float(self.c(z)), bounds=(x_s, self.params.x_right), method="bounded")
        p2 = float(res_p2.x)

        c_p1 = float(self.c(p1))
        c_p2 = float(self.c(p2))

        # Designate x_L as the lower peak (trap) and x_H as higher peak
        if c_p1 <= c_p2:
            x_L, x_H = p1, p2
        else:
            x_L, x_H = p2, p1

        return x_L, x_s, x_H

    def kramers_prefactor(self, sigma: Optional[float] = None, v: Optional[float] = None, dt: Optional[float] = None) -> float:
        """
        Computes Kramers prefactor tau_0 = 2*pi*sqrt(pi)*sigma / (v^2 * dt * sqrt(|c''(x_L)| * c''(x_s)))
        """
        sigma = self.params.sigma if sigma is None else sigma
        v = self.params.v if v is None else v
        dt = self.params.dt if dt is None else dt

        curv_product = np.abs(self.curv_xL) * self.curv_xs
        if curv_product <= 0:
            raise ValueError(f"Invalid extrema curvatures: c''(x_L)={self.curv_xL}, c''(x_s)={self.curv_xs}")

        tau_0 = (2.0 * np.pi * np.sqrt(np.pi) * sigma) / (v**2 * dt * np.sqrt(curv_product))
        return float(tau_0)

    def summary(self) -> Dict[str, Any]:
        return {
            "preset": self.preset_name,
            "x_L (lower peak)": self.x_L,
            "x_s (valley)": self.x_s,
            "x_H (higher peak)": self.x_H,
            "c(x_L)": self.c_xL,
            "c(x_s)": self.c_xs,
            "c(x_H)": self.c_xH,
            "Delta_U_trap": self.delta_u_trap,
            "Delta_c_target": self.delta_c_target,
            "c''(x_L)": self.curv_xL,
            "c''(x_s)": self.curv_xs,
            "c''(x_H)": self.curv_xH,
            "tau_0 (sigma=0.10)": self.kramers_prefactor(sigma=0.10)
        }


if __name__ == "__main__":
    print("=== Step 1: Landscape Geometry (Roadmap Version 2) ===")
    for name in ["L1a", "L1b", "L1c", "L2", "L3"]:
        ls = ChemicalLandscape.from_preset(name)
        s = ls.summary()
        print(f"\nPreset [{name}]:")
        for k, v in s.items():
            print(f"  {k:22s}: {v:12.6f}" if isinstance(v, float) else f"  {k:22s}: {v}")
