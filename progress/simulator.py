"""
Step 2, 3, 4, 5: Vectorized Monte Carlo Langevin Simulator.
Supports Independent Readings and Single-Sensor (Shared Reading) modes,
computes empirical MFPT, first-passage time distributions, drift, diffusivity,
action autocorrelations, and stochastic calculus convention estimators.
"""
from typing import Optional, Dict, Any
import numpy as np
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape


class VectorizedSimulator:
    def __init__(self, landscape: ChemicalLandscape, params: Optional[PhysicalParameters] = None):
        self.ls = landscape
        self.params: PhysicalParameters = landscape.params if params is None else params

    def simulate_mfpt(self, n_trajectories: Optional[int] = None, mode: str = "independent",
                      x_start: Optional[float] = None, absorbing_x: Optional[float] = None,
                      max_steps: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulates n_trajectories until each reaches absorbing_x.
        Returns first passage times in steps and physical time, plus summary statistics.
        """
        n_traj: int = self.params.N_mc if n_trajectories is None else int(n_trajectories)
        m_steps: int = self.params.max_steps if max_steps is None else int(max_steps)
        x_st: float = self.ls.x1_peak if x_start is None else float(x_start)
        abs_x: float = (self.ls.x_min + self.params.absorbing_delta) if absorbing_x is None else float(absorbing_x)
        sd: int = self.params.seed if seed is None else int(seed)

        rng = np.random.default_rng(sd)

        # State arrays
        x = np.full(n_traj, x_st, dtype=np.float64)
        a = rng.choice([-1.0, 1.0], size=n_traj).astype(np.float64)
        active = np.ones(n_traj, dtype=bool)
        escape_steps = np.full(n_traj, m_steps, dtype=np.int64)

        dt = self.params.dt
        v = self.params.v
        D = self.params.D
        sigma = self.params.sigma
        x_left = self.params.x_left

        sqrt_2Ddt = np.sqrt(2.0 * D * dt) if D > 0 else 0.0

        c_prev_noisy = None
        if mode == "single_sensor":
            c_prev_noisy = np.asarray(self.ls.c(x), dtype=np.float64) + sigma * rng.standard_normal(n_traj)

        step = 0
        while np.any(active) and step < m_steps:
            step += 1
            idx_active = np.where(active)[0]
            n_active = len(idx_active)

            # 1. Update position for active trajectories
            x_prev = x[idx_active]
            a_curr = a[idx_active]

            disp = v * a_curr * dt
            if D > 0:
                disp += sqrt_2Ddt * rng.standard_normal(n_active)

            x_new = np.maximum(x_prev + disp, x_left)
            x[idx_active] = x_new

            # 2. Check absorbing condition
            escaped = (x_new >= abs_x)
            if np.any(escaped):
                esc_indices = idx_active[escaped]
                escape_steps[esc_indices] = step
                active[esc_indices] = False

            if not np.any(active):
                break

            # 3. Update actions for remaining active trajectories
            idx_rem = np.where(active)[0]
            n_rem = len(idx_rem)
            x_rem = x[idx_rem]
            a_rem = a[idx_rem]

            if mode == "independent":
                c_now = np.asarray(self.ls.c(x_rem), dtype=np.float64)
                c_was = np.asarray(self.ls.c(x_rem - a_rem * v * dt), dtype=np.float64)
                true_delta = c_now - c_was
                delta_noisy = true_delta + np.sqrt(2.0) * sigma * rng.standard_normal(n_rem)
                persist = (delta_noisy > 0.0)
                a[idx_rem] = np.where(persist, a_rem, -a_rem)

            elif mode == "single_sensor":
                c_curr_noisy = np.asarray(self.ls.c(x_rem), dtype=np.float64) + sigma * rng.standard_normal(n_rem)
                if c_prev_noisy is not None:
                    delta_noisy = c_curr_noisy - c_prev_noisy[idx_rem]
                    c_prev_noisy[idx_rem] = c_curr_noisy
                    persist = (delta_noisy > 0.0)
                    a[idx_rem] = np.where(persist, a_rem, -a_rem)

        escape_times = escape_steps * dt
        escaped_mask = (escape_steps < m_steps)
        escaped_times = escape_times[escaped_mask]

        mfpt_val = float(np.mean(escaped_times)) if len(escaped_times) > 0 else float("inf")
        std_err = float(np.std(escaped_times) / np.sqrt(len(escaped_times))) if len(escaped_times) > 1 else 0.0

        return {
            "mode": mode,
            "n_trajectories": n_traj,
            "n_escaped": int(np.sum(escaped_mask)),
            "escape_steps": escape_steps,
            "escape_times": escape_times,
            "mfpt": mfpt_val,
            "std_error": std_err,
            "fraction_escaped": float(np.mean(escaped_mask))
        }

    def measure_drift_and_diffusivity(self, x_grid: np.ndarray, n_samples_per_point: int = 50000,
                                      mode: str = "independent") -> Dict[str, Any]:
        """
        Measures localized one-step drift mu(x) and diffusivity Deff(x) across a spatial grid.
        Verifies [AD] Eqs. (18), (19) and single-sensor factors kappa_mu, kappa_D.
        """
        rng = np.random.default_rng(self.params.seed)
        v = self.params.v
        dt = self.params.dt
        sigma = self.params.sigma

        mu_measured = []
        deff_measured = []

        for x0 in x_grid:
            a_init = rng.choice([-1.0, 1.0], size=n_samples_per_point).astype(np.float64)

            if mode == "independent":
                c_now = float(self.ls.c(x0))
                c_was = float(self.ls.c(x0 - a_init[0] * v * dt))  # per action
                # Vectorized difference
                c_was_arr = np.asarray(self.ls.c(x0 - a_init * v * dt), dtype=np.float64)
                true_delta = c_now - c_was_arr
                delta_noisy = true_delta + np.sqrt(2.0) * sigma * rng.standard_normal(n_samples_per_point)
                a_next = np.where(delta_noisy > 0.0, a_init, -a_init)

            elif mode == "single_sensor":
                c_prev = np.asarray(self.ls.c(x0 - a_init * v * dt), dtype=np.float64) + sigma * rng.standard_normal(n_samples_per_point)
                c_now_noisy = float(self.ls.c(x0)) + sigma * rng.standard_normal(n_samples_per_point)
                delta_noisy = c_now_noisy - c_prev
                a_next = np.where(delta_noisy > 0.0, a_init, -a_init)

            mu = v * float(np.mean(a_next))
            deff = (v**2 * dt / 2.0) * float(np.var(a_next))
            mu_measured.append(mu)
            deff_measured.append(deff)

        return {
            "x_grid": x_grid,
            "mu_measured": np.array(mu_measured, dtype=np.float64),
            "deff_measured": np.array(deff_measured, dtype=np.float64)
        }

    def measure_action_autocorrelation(self, n_steps: int = 100000, n_trajectories: int = 100) -> np.ndarray:
        """
        Measures <a_t a_{t+k}> on a flat landscape (c(x) = const) to verify the
        Gaussian arcsine law prediction: <a_t a_{t+2}> = -1/3 for single-sensor mode.
        """
        rng = np.random.default_rng(self.params.seed)
        sigma = self.params.sigma

        a = rng.choice([-1.0, 1.0], size=n_trajectories).astype(np.float64)
        c_prev = sigma * rng.standard_normal(n_trajectories)

        a_history = np.zeros((n_trajectories, n_steps), dtype=np.float64)

        for t in range(n_steps):
            c_curr = sigma * rng.standard_normal(n_trajectories)
            delta = c_curr - c_prev
            c_prev = c_curr
            a = np.where(delta > 0.0, a, -a)
            a_history[:, t] = a

        lags = 10
        corrs = np.zeros(lags + 1, dtype=np.float64)
        for k in range(lags + 1):
            if k == 0:
                corrs[k] = 1.0
            else:
                a_t = a_history[:, :-k].flatten()
                a_tk = a_history[:, k:].flatten()
                corrs[k] = float(np.mean(a_t * a_tk))

        return corrs


if __name__ == "__main__":
    ls = ChemicalLandscape.from_preset("shallow")
    sim = VectorizedSimulator(ls)
    print("Running quick Monte Carlo test (1000 trajectories)...")
    res_indep = sim.simulate_mfpt(n_trajectories=1000, mode="independent")
    print(f"  Independent Readings MFPT: {res_indep['mfpt']:.2f} s (+/- {res_indep['std_error']:.2f})")
    res_single = sim.simulate_mfpt(n_trajectories=1000, mode="single_sensor")
    print(f"  Single Sensor MFPT:       {res_single['mfpt']:.2f} s (+/- {res_single['std_error']:.2f})")
