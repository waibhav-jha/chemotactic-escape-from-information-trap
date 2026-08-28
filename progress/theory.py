"""
Step 3, 4, 6, 7: Continuum Theory, Fokker-Planck Moments, Stable MFPT Double Quadrature,
and Saddle-Point Kramers Formula.
Implements the exact formulas derived in [AD] and [CH].
"""
from typing import Optional, Any
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
    # 0-Bit Memoryless Drift & Diffusivity ([AD] §2.1)
    # -------------------------------------------------------------
    def b0_drift(self, x: Any, theta: float = 0.5) -> Any:
        """0-bit threshold policy drift: mu(x) = v * (2*Phi((c(x)-theta)/sigma) - 1)."""
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
    # 1-Bit Temporal Comparison Drift & Diffusivity ([AD] §2.2, §4)
    # -------------------------------------------------------------
    def b1_snr(self, x: Any) -> Any:
        """Decision SNR s(x) from [AD] Eq. (13)."""
        dc_val = np.abs(np.asarray(self.ls.dc(x), dtype=np.float64))
        numerator = dc_val * self.params.v * self.params.dt
        denominator = np.sqrt(2.0 * self.params.sigma**2 + 2.0 * self.params.D * self.params.dt * dc_val**2)
        return numerator / denominator

    def b1_drift(self, x: Any, mode: str = "independent") -> Any:
        """
        1-bit temporal comparison drift:
        Independent: mu(x) = v * sign(c') * erf(|c'|*v*dt / (2*sigma)) ([AD] Eq. 18)
        Single-sensor: scaled by kappa_mu = 0.7208 ([AD] Eq. 449)
        """
        dc_val = np.asarray(self.ls.dc(x), dtype=np.float64)
        z = np.abs(dc_val) * self.params.v * self.params.dt / (2.0 * self.params.sigma)
        mu_indep = self.params.v * np.sign(dc_val) * erf(z)
        if mode == "single_sensor":
            return 0.7208 * mu_indep
        return mu_indep

    def b1_deff(self, x: Any, mode: str = "independent") -> Any:
        """
        1-bit effective diffusivity:
        Independent: Deff(x) = D + 2*v^2*dt * Phi(s)*(1 - Phi(s)) ([AD] Eq. 19)
        Single-sensor: Green-Kubo corrected by factor kappa_D = 0.523 ([AD] Eq. 451)
        """
        s = self.b1_snr(x)
        phi_s = ndtr(s)
        deff_indep = self.params.D + 2.0 * (self.params.v**2 * self.params.dt) * phi_s * (1.0 - phi_s)
        if mode == "single_sensor":
            return self.params.D + 0.523 * (deff_indep - self.params.D)
        return deff_indep

    # -------------------------------------------------------------
    # Effective Potential & Temperature in Weak-Signal Regime ([AD] §3)
    # -------------------------------------------------------------
    def effective_potential(self, x: Any) -> Any:
        """U_eff(x) = - (v^2 * dt / (sqrt(pi) * sigma)) * c(x)."""
        coeff = (self.params.v**2 * self.params.dt) / (np.sqrt(np.pi) * self.params.sigma)
        return - coeff * np.asarray(self.ls.c(x), dtype=np.float64)

    def effective_temperature(self) -> float:
        """T_eff = D + 0.5 * v^2 * dt."""
        return float(self.params.D + 0.5 * self.params.v**2 * self.params.dt)

    def delta_u_eff(self) -> float:
        """Delta U_eff = U_eff(x_min) - U_eff(x1) = (v^2 * dt / (sqrt(pi) * sigma)) * Delta U_trap."""
        coeff = (self.params.v**2 * self.params.dt) / (np.sqrt(np.pi) * self.params.sigma)
        return float(coeff * self.ls.delta_u_trap)

    # -------------------------------------------------------------
    # Stable 1D MFPT Double Quadrature ([AD] §6, [CH] §7)
    # -------------------------------------------------------------
    def mfpt_quadrature(self, x_start: Optional[float] = None, r_bound: Optional[float] = None,
                        b_bound: Optional[float] = None, nx: int = 4001, mode: str = "independent") -> float:
        """
        Evaluates the exact 1D Fokker-Planck MFPT double integral from [AD] Eq. (36):
        T(x) = int_x^b dy int_r^y dz exp(Phi(y) - Phi(z)) / Deff(z)
        where Phi(y) = - int_r^y (mu(z) / Deff(z)) dz.
        Uses log-difference stabilization exp(Phi(y) - Phi(z)) to prevent overflow.
        """
        x_start_val: float = self.ls.x1_peak if x_start is None else float(x_start)
        r_bound_val: float = self.params.x_left if r_bound is None else float(r_bound)
        b_bound_val: float = (self.ls.x_min + self.params.absorbing_delta) if b_bound is None else float(b_bound)

        # Grid on [r, b]
        grid = np.linspace(r_bound_val, b_bound_val, nx, dtype=np.float64)
        mu_vals = np.asarray(self.b1_drift(grid, mode=mode), dtype=np.float64)
        deff_vals = np.asarray(self.b1_deff(grid, mode=mode), dtype=np.float64)

        # Integrand for potential Phi(y) = - int_r^y (mu/Deff) dz
        integrand_phi = - (mu_vals / deff_vals)
        phi_vals = cumulative_trapezoid(integrand_phi, grid, initial=0.0)

        # Stable inner integral: I(y) = int_r^y exp(Phi(y) - Phi(z)) / Deff(z) dz
        inner_vals = np.zeros(nx, dtype=np.float64)
        for i in range(1, nx):
            z_sub = grid[:i+1]
            exp_diff = np.exp(phi_vals[i] - phi_vals[:i+1])
            inner_vals[i] = simpson(exp_diff / deff_vals[:i+1], x=z_sub)

        # Outer integral: T(x) = int_x^b inner(y) dy
        idx_start = int(np.searchsorted(grid, x_start_val))
        if idx_start >= nx - 1:
            return 0.0

        outer_grid = grid[idx_start:]
        outer_integrand = inner_vals[idx_start:]
        t_mfpt = simpson(outer_integrand, x=outer_grid)
        return float(t_mfpt)

    # -------------------------------------------------------------
    # Saddle-Point Kramers Formula ([AD] §3.2, Eq. 37)
    # -------------------------------------------------------------
    def kramers_mfpt(self, mode: str = "independent") -> float:
        """
        Asymptotic Kramers escape time:
        T_saddle = tau_0 * exp(Delta U_eff / T_eff)
        """
        tau_0 = self.ls.kramers_prefactor(sigma=self.params.sigma, v=self.params.v, dt=self.params.dt)
        delta_u_eff = self.delta_u_eff()
        t_eff = self.effective_temperature()
        exponent = delta_u_eff / t_eff
        if mode == "single_sensor":
            exponent *= (0.7208 / 0.523)
        return float(tau_0 * np.exp(exponent))


if __name__ == "__main__":
    ls = ChemicalLandscape.from_preset("medium")
    theory = ContinuumTheory(ls)
    t_quad = theory.mfpt_quadrature()
    t_saddle = theory.kramers_mfpt()
    print(f"Preset Medium (Delta U={ls.delta_u_trap:.4f}, sigma={ls.params.sigma}):")
    print(f"  Fokker-Planck Double Quadrature MFPT: {t_quad:12.4f}")
    print(f"  Asymptotic Kramers Saddle-Point MFPT: {t_saddle:12.4f}")
    print(f"  Ratio (Quadrature / Kramers):         {t_quad / t_saddle:12.4f}")
