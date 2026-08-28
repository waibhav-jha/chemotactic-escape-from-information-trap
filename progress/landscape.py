"""
Step 1: Chemical Landscape & Barrier Geometry.
Constructs c(x), exact derivatives c'(x) and c''(x), finds extrema (x1, x_min, x2),
computes trap barrier Delta U_trap, and evaluates Kramers curvatures.
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize_scalar
from progress.parameters import PhysicalParameters, LANDSCAPE_PRESETS


class ChemicalLandscape:
    def __init__(self, params: Optional[PhysicalParameters] = None):
        if params is None:
            params = PhysicalParameters()
        self.params = params
        self.x1_center: float = params.x1
        self.x2_center: float = params.x2
        self.a1: float = params.a1
        self.a2: float = params.a2
        self.s1: float = params.s1
        self.s2: float = params.s2

        # Find exact numerical extrema
        self.x1_peak, self.x_min, self.x2_peak = self._find_extrema()
        self.c_x1: float = float(self.c(self.x1_peak))
        self.c_xmin: float = float(self.c(self.x_min))
        self.c_x2: float = float(self.c(self.x2_peak))
        self.delta_u_trap: float = self.c_x1 - self.c_xmin

        # Curvatures
        self.curv_x1: float = float(self.d2c(self.x1_peak))     # Negative for peak
        self.curv_xmin: float = float(self.d2c(self.x_min))     # Positive for minimum

    @classmethod
    def from_preset(cls, preset_name: str, **override_params) -> "ChemicalLandscape":
        preset = LANDSCAPE_PRESETS[preset_name]
        p = PhysicalParameters(
            x1=preset["x1"],
            x2=preset["x2"],
            a1=preset["a1"],
            a2=preset["a2"],
            s1=preset["s1"],
            s2=preset["s2"],
            **override_params
        )
        return cls(p)

    def c(self, x: Any) -> Any:
        """Chemical concentration c(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        g1 = self.a1 * np.exp(- (x_arr - self.x1_center)**2 / (2.0 * self.s1**2))
        g2 = self.a2 * np.exp(- (x_arr - self.x2_center)**2 / (2.0 * self.s2**2))
        return g1 + g2

    def dc(self, x: Any) -> Any:
        """First spatial derivative c'(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        t1 = - self.a1 * (x_arr - self.x1_center) / (self.s1**2) * np.exp(- (x_arr - self.x1_center)**2 / (2.0 * self.s1**2))
        t2 = - self.a2 * (x_arr - self.x2_center) / (self.s2**2) * np.exp(- (x_arr - self.x2_center)**2 / (2.0 * self.s2**2))
        return t1 + t2

    def d2c(self, x: Any) -> Any:
        """Second spatial derivative (curvature) c''(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        g1 = np.exp(- (x_arr - self.x1_center)**2 / (2.0 * self.s1**2))
        g2 = np.exp(- (x_arr - self.x2_center)**2 / (2.0 * self.s2**2))
        t1 = (self.a1 / self.s1**2) * (((x_arr - self.x1_center)**2 / self.s1**2) - 1.0) * g1
        t2 = (self.a2 / self.s2**2) * (((x_arr - self.x2_center)**2 / self.s2**2) - 1.0) * g2
        return t1 + t2

    def _find_extrema(self) -> Tuple[float, float, float]:
        """Locates the local maximum x1, valley minimum x_min, and maximum x2."""
        # Find x_min between x1_center and x2_center
        res_min = minimize_scalar(lambda z: float(self.c(z)), bounds=(self.x1_center, self.x2_center), method="bounded")
        x_min = float(res_min.x)

        # Find x1_peak near x1_center (to the left of x_min)
        res_x1 = minimize_scalar(lambda z: -float(self.c(z)), bounds=(self.params.x_left, x_min), method="bounded")
        x1_peak = float(res_x1.x)

        # Find x2_peak near x2_center (to the right of x_min)
        res_x2 = minimize_scalar(lambda z: -float(self.c(z)), bounds=(x_min, self.params.x_right), method="bounded")
        x2_peak = float(res_x2.x)

        return x1_peak, x_min, x2_peak

    def kramers_prefactor(self, sigma: Optional[float] = None, v: Optional[float] = None, dt: Optional[float] = None) -> float:
        """
        Computes Kramers prefactor tau_0 = 2*pi*sqrt(pi)*sigma / (v^2 * dt * sqrt(|c''(x1)| * c''(x_min)))
        From [AD] Eq. (407).
        """
        sigma = self.params.sigma if sigma is None else sigma
        v = self.params.v if v is None else v
        dt = self.params.dt if dt is None else dt

        curv_product = np.abs(self.curv_x1) * self.curv_xmin
        if curv_product <= 0:
            raise ValueError(f"Invalid extrema curvatures: c''(x1)={self.curv_x1}, c''(xmin)={self.curv_xmin}")

        tau_0 = (2.0 * np.pi * np.sqrt(np.pi) * sigma) / (v**2 * dt * np.sqrt(curv_product))
        return float(tau_0)

    def summary(self) -> Dict[str, Any]:
        return {
            "x1_peak": self.x1_peak,
            "x_min": self.x_min,
            "x2_peak": self.x2_peak,
            "c(x1)": self.c_x1,
            "c(x_min)": self.c_xmin,
            "c(x2)": self.c_x2,
            "Delta_U_trap": self.delta_u_trap,
            "c''(x1)": self.curv_x1,
            "c''(x_min)": self.curv_xmin,
            "tau_0 (sigma=0.10)": self.kramers_prefactor(sigma=0.10)
        }


if __name__ == "__main__":
    print("=== Step 1: Landscape Presets Summary ===")
    for name in ["shallow", "medium", "deep"]:
        ls = ChemicalLandscape.from_preset(name)
        s = ls.summary()
        print(f"\nPreset [{name.upper()}]:")
        for k, v in s.items():
            print(f"  {k:20s}: {v:12.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")
