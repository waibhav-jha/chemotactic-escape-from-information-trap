"""
Continuum Theory, Stochastic Calculus Conventions, MFPT Double Quadrature,
and Kramers Deep-Barrier Asymptotics (Roadmap Version 2: Steps 3, 5, 7, 8, 10).
Implements exact analytical formulas for 0-bit and 1-bit policies, weak-signal effective
potentials, validity checklists, and event-resolved first-passage predictions (E1, E2, E3).
"""
import os
import sys
from typing import Optional, Any, Dict

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.special import erf, ndtr
from scipy.integrate import cumulative_trapezoid, simpson
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape


class ContinuumTheory:
    def __init__(self, landscape: ChemicalLandscape, params: Optional[PhysicalParameters] = None):
        self.ls = landscape
        self.params: PhysicalParameters = landscape.params if params is None else params

    # -------------------------------------------------------------
    # 0-Bit Memoryless Drift & Diffusivity (Step 3)
    # -------------------------------------------------------------
    def b0_drift(self, x: Any, theta: float = 0.5) -> Any:
        """0-bit threshold policy drift: mu_0(x) = v * (2*Phi((c(x)-theta)/sigma) - 1)."""
        c_val = np.asarray(self.ls.c(x), dtype=np.float64)
        f_val = 2.0 * ndtr((c_val - theta) / self.params.sigma) - 1.0
        return self.params.v * f_val

    def b0_deff(self, x: Any, theta: float = 0.5) -> Any:
        """0-bit effective diffusivity: Deff(x) = D + 0.5 * v^2 * dt * (1 - F(c)^2)."""
        c_val = np.asarray(self.ls.c(x), dtype=np.float64)
        f_val = 2.0 * ndtr((c_val - theta) / self.params.sigma) - 1.0
        g_val = 0.5 * (1.0 - f_val**2)
        return self.params.D + (self.params.v**2 * self.params.dt) * g_val

    # -------------------------------------------------------------
    # 1-Bit Temporal Comparison Drift & Diffusivity (Step 3)
    # -------------------------------------------------------------
    def b1_snr(self, x: Any) -> Any:
        """Decision SNR s(x) = |c'(x)| * v * dt_s / (sqrt(2) * sigma)."""
        dc_val = np.abs(np.asarray(self.ls.dc(x), dtype=np.float64))
        numerator = dc_val * self.params.v * self.params.dt_s
        denominator = np.sqrt(2.0 * self.params.sigma**2 + 2.0 * self.params.D * self.params.dt * dc_val**2)
        return numerator / denominator

    def b1_drift(self, x: Any, mode: str = "independent", alpha: Optional[float] = None) -> Any:
        """
        1-bit temporal comparison drift:
        Independent (Architecture I): mu_1(x) = v * sign(c') * erf(|c'|*v*dt_s / (2*sigma))
        Single-sensor (Architecture II): scaled by kappa_mu = 0.7208
        Applies Itô correction if alpha < 1.0: mu_Ito = mu_ref + (1 - alpha) * D'_eff(x)
        """
        dc_val = np.asarray(self.ls.dc(x), dtype=np.float64)
        z = np.abs(dc_val) * self.params.v * self.params.dt_s / (2.0 * self.params.sigma)
        mu_indep = self.params.v * np.sign(dc_val) * erf(z)
        mu_ref = 0.7208 * mu_indep if mode == "single_sensor" else mu_indep

        # Check stochastic calculus convention
        conv = self.params.alpha_conv if alpha is None else alpha
        if np.isclose(conv, 1.0):
            return mu_ref
        
        # Itô drift with spurious drift term: (1 - alpha) * d(Deff)/dx
        eps = 1e-4
        deff_plus = self.b1_deff(x + eps, mode=mode)
        deff_minus = self.b1_deff(x - eps, mode=mode)
        d_deff_dx = (deff_plus - deff_minus) / (2.0 * eps)
        return mu_ref + (1.0 - conv) * d_deff_dx

    def b1_deff(self, x: Any, mode: str = "independent") -> Any:
        """
        1-bit effective diffusivity:
        Independent: Deff(x) = D + 2*v^2*dt_s * Phi(s)*(1 - Phi(s))
        Single-sensor: Green-Kubo corrected by factor kappa_D = 0.523
        """
        s = self.b1_snr(x)
        phi_s = ndtr(s)
        deff_indep = self.params.D + 2.0 * (self.params.v**2 * self.params.dt_s) * phi_s * (1.0 - phi_s)
        if mode == "single_sensor":
            return self.params.D + 0.523 * (deff_indep - self.params.D)
        return deff_indep

    # -------------------------------------------------------------
    # Weak-Signal Effective Potential & Validity Regime (Step 5)
    # -------------------------------------------------------------
    def effective_potential(self, x: Any) -> Any:
        """V(x) = - A * c(x), where A = v^2 * dt_s / (sqrt(pi) * sigma)."""
        a_const = (self.params.v**2 * self.params.dt_s) / (np.sqrt(np.pi) * self.params.sigma)
        return - a_const * np.asarray(self.ls.c(x), dtype=np.float64)

    def effective_temperature(self) -> float:
        """D* = D + 0.5 * v^2 * dt_s."""
        return float(self.params.D + 0.5 * self.params.v**2 * self.params.dt_s)

    def delta_v_eff(self) -> float:
        """Delta V = V(x_s) - V(x_L) = A * Delta U_trap."""
        a_const = (self.params.v**2 * self.params.dt_s) / (np.sqrt(np.pi) * self.params.sigma)
        return float(a_const * self.ls.delta_u_trap)

    def check_validity_regime(self) -> Dict[str, Any]:
        """
        Validates the 4 conditions for the 2/sqrt(pi) Arrhenius escape law (Step 5):
        1. Weak sensing: v*dt_s * max|c'| / (sqrt(2)*sigma) << 1
        2. Deep effective barrier: Delta V / D* >> 1
        3. Local-step approximation: v*dt_s / ell_c << 1
        4. Small Brownian contribution: D << v^2*dt_s / 2
        """
        x_grid = np.linspace(self.ls.x_L, self.ls.x_s, 500)
        max_grad = float(np.max(np.abs(self.ls.dc(x_grid))))
        weak_sensing = (self.params.v * self.params.dt_s * max_grad) / (np.sqrt(2.0) * self.params.sigma)
        
        delta_v = self.delta_v_eff()
        d_star = self.effective_temperature()
        deep_barrier = delta_v / d_star

        ell_c = float(self.ls.s1) # Characteristic landscape width
        local_step = (self.params.v * self.params.dt_s) / ell_c

        brownian_ratio = self.params.D / max(1e-12, 0.5 * self.params.v**2 * self.params.dt_s)

        return {
            "weak_sensing_param": weak_sensing,
            "weak_sensing_valid": weak_sensing < 0.35,
            "deep_barrier_param": deep_barrier,
            "deep_barrier_valid": deep_barrier > 2.0,
            "local_step_param": local_step,
            "local_step_valid": local_step < 0.10,
            "brownian_ratio": brownian_ratio,
            "brownian_valid": brownian_ratio < 0.10,
            "asymptotic_slope_target": 2.0 / np.sqrt(np.pi) # 1.12838
        }

    # -------------------------------------------------------------
    # 1D MFPT Double Quadrature across Events E1, E2, E3 (Step 8)
    # -------------------------------------------------------------
    def mfpt_quadrature(self, event: str = "E1", x_start: Optional[float] = None,
                        r_bound: Optional[float] = None, nx: int = 4001, mode: str = "independent") -> float:
        """
        Evaluates 1D Fokker-Planck MFPT double integral:
        T(x) = int_x^B dy int_r^y dz exp(Phi(y) - Phi(z)) / Deff(z)
        where Phi(y) = - int_r^y (mu(z) / Deff(z)) dz.
        Absorbing boundary B is determined by event:
          - E1: B = x_s (valley saddle)
          - E2: B = x_s + delta
          - E3: B = x_H - target_radius (target basin boundary)
        """
        x_start_val = self.ls.x_L if x_start is None else float(x_start)
        r_bound_val = self.params.x_left if r_bound is None else float(r_bound)

        if event == "E1":
            b_bound_val = float(self.ls.x_s)
        elif event == "E2":
            b_bound_val = float(self.ls.x_s + max(0.05, self.params.absorbing_delta))
        elif event == "E3":
            b_bound_val = float(self.ls.x_H - self.params.target_radius)
        else:
            b_bound_val = float(self.ls.x_s)

        if x_start_val >= b_bound_val:
            return 0.0

        grid = np.linspace(r_bound_val, b_bound_val, nx, dtype=np.float64)
        mu_vals = np.asarray(self.b1_drift(grid, mode=mode), dtype=np.float64)
        deff_vals = np.asarray(self.b1_deff(grid, mode=mode), dtype=np.float64)

        # Integrating factor Phi(y) = - int_r^y (mu / Deff) dz
        integrand_phi = - (mu_vals / deff_vals)
        phi_vals = cumulative_trapezoid(integrand_phi, grid, initial=0.0)

        # Stable inner integral
        inner_vals = np.zeros(nx, dtype=np.float64)
        for i in range(1, nx):
            z_sub = grid[:i+1]
            exp_diff = np.exp(phi_vals[i] - phi_vals[:i+1])
            inner_vals[i] = simpson(exp_diff / deff_vals[:i+1], x=z_sub)

        idx_start = int(np.searchsorted(grid, x_start_val))
        if idx_start >= nx - 1:
            return 0.0

        outer_grid = grid[idx_start:]
        outer_integrand = inner_vals[idx_start:]
        return float(simpson(outer_integrand, x=outer_grid))

    # -------------------------------------------------------------
    # Kramers Deep-Barrier Asymptotics across Events (Step 10)
    # -------------------------------------------------------------
    def kramers_mfpt(self, event: str = "E1", mode: str = "independent") -> float:
        """
        Asymptotic Kramers escape time across events:
        T_K = tau_0 * exp(Delta V / D*)
        where Delta V = V(x_s) - V(x_L).
        """
        tau_0 = self.ls.kramers_prefactor(sigma=self.params.sigma, v=self.params.v, dt=self.params.dt)
        delta_v = self.delta_v_eff()
        d_star = self.effective_temperature()
        exponent = delta_v / d_star

        if mode == "single_sensor":
            exponent *= (0.7208 / 0.523)

        base_t = tau_0 * np.exp(exponent)

        # Recrossing corrections for E2 and E3
        if event == "E1":
            return float(base_t)
        elif event == "E2":
            # Valley crossing requires crossing the saddle region; typically ~2x saddle encounter time
            return float(base_t * 1.5)
        elif event == "E3":
            # Basin-to-basin transition: incorporates downhill travel to x_H
            t_downhill = abs(self.ls.x_H - self.ls.x_s) / max(0.1, self.params.v)
            return float(base_t * 2.0 + t_downhill)
        return float(base_t)


if __name__ == "__main__":
    ls = ChemicalLandscape.from_preset("L1b")
    theory = ContinuumTheory(ls)
    regime = theory.check_validity_regime()
    print("=== Step 5: Validity Regime Checklist ===")
    for k, v in regime.items():
        print(f"  {k:28s}: {v}")

    print("\n=== Step 8 & 10: Event-Resolved MFPT Predictions ===")
    for ev in ["E1", "E2", "E3"]:
        t_q = theory.mfpt_quadrature(event=ev)
        t_k = theory.kramers_mfpt(event=ev)
        print(f"  Event [{ev}]: Quadrature = {t_q:10.2f} s | Kramers = {t_k:10.2f} s | Ratio = {t_q/max(1e-6, t_k):.4f}")
