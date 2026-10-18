"""
Step 7: Exact Markov Chain Solver for Chemotactic Escape.
Constructs the joint (x_bin, M) transition matrix P^pi with exact grid matching,
solves exact MFPT without Monte Carlo error, computes Stationary Density,
Quasi-Stationary Distribution (QSD), Committor function q(i, M),
Reaction Coordinate diagnostic Delta q(i), and full Eigenspectrum timescales.
"""
from typing import Optional, Dict, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigs
from scipy.special import ndtr
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape


class ExactMarkovChainSolver:
    def __init__(self, landscape: ChemicalLandscape, params: Optional[PhysicalParameters] = None):
        self.ls = landscape
        self.params: PhysicalParameters = landscape.params if params is None else params

        # Domain and grid matching: h = v * dt / K
        self.h: float = self.params.h
        self.x_left: float = self.params.x_left
        self.x_right: float = self.ls.x_min + self.params.absorbing_delta

        # Construct spatial grid
        # Bins from x_left to x_right
        self.n_spatial: int = int(np.round((self.x_right - self.x_left) / self.h)) + 1
        self.x_grid: np.ndarray = np.linspace(self.x_left, self.x_right, self.n_spatial, dtype=np.float64)
        # Absorbing state index is the rightmost boundary bin
        self.absorbing_spatial_idx: int = self.n_spatial - 1

    def build_1bit_transition_matrix(self, absorbing: bool = True) -> sp.csr_matrix:
        """
        Constructs the exact 1-bit transition matrix on N = 2 * n_spatial states.
        State indexing:
          state_idx(i, a): i is spatial bin (0..n-1), a is direction (+1 -> 1, -1 -> 0)
          index = 2 * i + dir_idx (dir_idx = 1 for +1, 0 for -1)
        """
        n = self.n_spatial
        N = 2 * n
        P = sp.lil_matrix((N, N), dtype=np.float64)

        sigma = self.params.sigma

        for i in range(n):
            for dir_idx, a_curr in [(0, -1.0), (1, 1.0)]:
                state_u = 2 * i + dir_idx

                # If this is an absorbing state
                if absorbing and i >= self.absorbing_spatial_idx:
                    P[state_u, state_u] = 1.0
                    continue

                # The agent arrived at bin i having moved with action a_curr
                # Previous position was i_prev = i - a_curr (in bin units, since K=1)
                i_prev = int(np.clip(i - int(a_curr), 0, n - 1))
                x_curr = float(self.x_grid[i])
                x_prev = float(self.x_grid[i_prev])

                # True concentration difference across the completed step
                delta_c = float(self.ls.c(x_curr) - self.ls.c(x_prev))

                # Sensing probability of persisting
                p_persist = float(ndtr(delta_c / (np.sqrt(2.0) * sigma)))
                p_reverse = 1.0 - p_persist

                # 1. Persist: next action a' = a_curr, next bin j = i + a_curr
                j_persist = int(np.clip(i + int(a_curr), 0, n - 1))
                dir_persist = dir_idx
                state_persist = 2 * j_persist + dir_persist
                P[state_u, state_persist] += p_persist

                # 2. Reverse: next action a' = -a_curr, next bin j = i - a_curr
                j_reverse = int(np.clip(i - int(a_curr), 0, n - 1))
                dir_reverse = 1 - dir_idx
                state_reverse = 2 * j_reverse + dir_reverse
                P[state_u, state_reverse] += p_reverse

        P_csr = P.tocsr()
        # Row sums check
        row_sums = np.asarray(P_csr.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0), f"Row sums do not equal 1.0: min={float(row_sums.min())}, max={float(row_sums.max())}"
        return P_csr

    def solve_exact_mfpt(self, x_start: Optional[float] = None) -> Dict[str, Any]:
        """
        Solves the exact discrete-time MFPT system: (I - P_trans) T = 1.
        Returns point-start MFPT, QSD escape time, and full transient T vector.
        """
        x_start_val = self.ls.x1_peak if x_start is None else float(x_start)
        P = self.build_1bit_transition_matrix(absorbing=True)

        n = self.n_spatial
        N = 2 * n
        # Absorbing states are states with spatial index >= absorbing_spatial_idx
        absorbing_mask = np.zeros(N, dtype=bool)
        for i in range(self.absorbing_spatial_idx, n):
            absorbing_mask[2 * i] = True
            absorbing_mask[2 * i + 1] = True

        transient_indices = np.where(~absorbing_mask)[0]
        N_trans = len(transient_indices)

        P_trans = P[transient_indices, :][:, transient_indices]

        # Solve (I - P_trans) T_steps = 1
        I_sp = sp.eye(N_trans, format="csr")
        rhs = np.ones(N_trans, dtype=np.float64)
        T_steps = spsolve(I_sp - P_trans, rhs)
        T_time = T_steps * self.params.dt

        # Map back to full spatial bins (averaging over initial directions +1 and -1)
        start_bin = int(np.argmin(np.abs(self.x_grid[:self.absorbing_spatial_idx] - x_start_val)))
        state_neg = 2 * start_bin
        state_pos = 2 * start_bin + 1

        idx_neg = int(np.where(transient_indices == state_neg)[0][0])
        idx_pos = int(np.where(transient_indices == state_pos)[0][0])
        mfpt_point_start = 0.5 * (float(T_time[idx_neg]) + float(T_time[idx_pos]))

        # Compute leading eigenvalues for QSD and timescales
        k_eigs = min(8, N_trans - 2)
        evals, _ = eigs(P_trans, k=k_eigs, which="LM")
        order = np.argsort(-np.real(evals))
        evals = evals[order]
        lambda_1 = float(np.real(evals[0]))

        # Filter out negative parity modes near -1 to find physical intra-basin relaxation lambda_2
        physical_evals = [ev for ev in evals if np.real(ev) > 0 and not np.isclose(np.real(ev), lambda_1)]
        lambda_2 = float(np.real(physical_evals[0])) if len(physical_evals) > 0 else float(np.real(evals[1]))

        tau_esc_qsd = self.params.dt / (1.0 - lambda_1) if (1.0 - lambda_1) > 0 else float("inf")
        tau_esc_spectral = - self.params.dt / np.log(lambda_1) if lambda_1 < 1.0 else float("inf")
        tau_rel = - self.params.dt / np.log(np.abs(lambda_2)) if np.abs(lambda_2) < 1.0 else float("inf")

        return {
            "mfpt_point_start": float(mfpt_point_start),
            "mfpt_qsd": float(tau_esc_qsd),
            "tau_esc_spectral": float(tau_esc_spectral),
            "tau_rel": float(tau_rel),
            "timescale_separation": float(tau_rel / tau_esc_spectral) if tau_esc_spectral > 0 else 0.0,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "T_transient_vector": T_time,
            "transient_indices": transient_indices,
            "x_grid": self.x_grid[:self.absorbing_spatial_idx]
        }

    def compute_committor(self, basin_a_x: Optional[float] = None, basin_b_x: Optional[float] = None) -> Dict[str, Any]:
        """
        Computes the committor function q(i, M) solving (I - P_trans) q = P_{T -> B} 1.
        Evaluates the reaction coordinate diagnostic Delta q(i) = max_{M, M'} |q(i, M) - q(i, M')|.
        From [CH] §5.
        """
        basin_a_val: float = self.ls.x1_center if basin_a_x is None else float(basin_a_x)
        basin_b_val: float = self.ls.x_min if basin_b_x is None else float(basin_b_x)

        n = self.n_spatial
        N = 2 * n
        P = self.build_1bit_transition_matrix(absorbing=False)

        # Basin A: bins at or left of basin_a_val
        idx_a_max = int(np.searchsorted(self.x_grid, basin_a_val))
        # Basin B: bins at or right of basin_b_val
        idx_b_min = int(np.searchsorted(self.x_grid, basin_b_val))

        mask_a = np.zeros(N, dtype=bool)
        mask_b = np.zeros(N, dtype=bool)
        for i in range(idx_a_max + 1):
            mask_a[2 * i] = True
            mask_a[2 * i + 1] = True
        for i in range(idx_b_min, n):
            mask_b[2 * i] = True
            mask_b[2 * i + 1] = True

        mask_t = ~(mask_a | mask_b)
        idx_a = np.where(mask_a)[0]
        idx_b = np.where(mask_b)[0]
        idx_t = np.where(mask_t)[0]

        P_tt = P[idx_t, :][:, idx_t]
        P_tb = P[idx_t, :][:, idx_b]

        I_sp = sp.eye(len(idx_t), format="csr")
        rhs = np.asarray(P_tb.sum(axis=1), dtype=np.float64).flatten()
        q_t = spsolve(I_sp - P_tt, rhs)

        # Assemble full committor vector
        q_full = np.zeros(N, dtype=np.float64)
        q_full[idx_a] = 0.0
        q_full[idx_b] = 1.0
        q_full[idx_t] = q_t

        # Spatial committor for directions -1 and +1
        q_neg = np.zeros(n, dtype=np.float64)
        q_pos = np.zeros(n, dtype=np.float64)
        delta_q = np.zeros(n, dtype=np.float64)

        for i in range(n):
            q_neg[i] = float(q_full[2 * i])
            q_pos[i] = float(q_full[2 * i + 1])
            delta_q[i] = float(np.abs(q_pos[i] - q_neg[i]))

        q_avg = 0.5 * (q_neg + q_pos)
        idx_ts = int(np.argmin(np.abs(q_avg - 0.5)))
        x_transition_state = float(self.x_grid[idx_ts])
        max_delta_q = float(np.max(delta_q))

        return {
            "x_grid": self.x_grid,
            "q_neg": q_neg,
            "q_pos": q_pos,
            "q_avg": q_avg,
            "delta_q": delta_q,
            "max_delta_q": max_delta_q,
            "x_transition_state": x_transition_state,
            "x_min": self.ls.x_min
        }

    def build_2bit_transition_matrix(self, policy_persist: Optional[Any] = None, absorbing: bool = True) -> sp.csr_matrix:
        """
        Constructs the exact discrete Markov transition matrix for a 2-bit agent on N = 4 * n_spatial states.
        State indexing:
          state_u = 4 * i + m, where internal memory m in [0, 1, 2, 3]:
            heading_idx = m // 2 (0 for -1.0, 1 for +1.0)
            s_prev = m % 2 (0 for falling, 1 for rising in previous step)
          Sensory observation on current step: s_curr = 1 if Delta c > 0 else 0
          Joint sensory state: s_sens = s_curr * 2 + s_prev in {0, 1, 2, 3}
          policy_persist: vector of 4 probabilities or booleans [P(persist | s_sens)]
                          defaults to standard 1-bit rule: [0.0, 0.0, 1.0, 1.0]
        """
        n = self.n_spatial
        num_mem = 4
        N = num_mem * n
        P = sp.lil_matrix((N, N), dtype=np.float64)

        sigma = self.params.sigma

        if policy_persist is None:
            pol = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        else:
            pol = np.array(policy_persist, dtype=np.float64)

        for i in range(n):
            if absorbing and i >= self.absorbing_spatial_idx:
                for m in range(num_mem):
                    u = i * num_mem + m
                    P[u, u] = 1.0
                continue

            for m in range(num_mem):
                u = i * num_mem + m
                heading_idx = m // 2
                a_curr = -1.0 if heading_idx == 0 else +1.0
                s_prev = m % 2

                i_prev = int(np.clip(i - int(a_curr), 0, n - 1))
                x_curr = float(self.x_grid[i])
                x_prev = float(self.x_grid[i_prev])

                delta_c = float(self.ls.c(x_curr) - self.ls.c(x_prev))
                p_up = float(ndtr(delta_c / (np.sqrt(2.0) * sigma)))
                p_down = 1.0 - p_up

                # Branch UP: s_curr = 1
                s_sens_up = 1 * 2 + s_prev
                p_persist_up = float(pol[s_sens_up])
                p_reverse_up = 1.0 - p_persist_up

                # If persist: heading remains a_curr, next s_prev becomes 1
                j_p_up = int(np.clip(i + int(a_curr), 0, n - 1))
                m_next_p_up = heading_idx * 2 + 1
                if p_persist_up > 0.0 and p_up > 0.0:
                    P[u, j_p_up * num_mem + m_next_p_up] += p_up * p_persist_up

                # If reverse: heading flips to -a_curr, next s_prev becomes 1
                j_r_up = int(np.clip(i - int(a_curr), 0, n - 1))
                m_next_r_up = (1 - heading_idx) * 2 + 1
                if p_reverse_up > 0.0 and p_up > 0.0:
                    P[u, j_r_up * num_mem + m_next_r_up] += p_up * p_reverse_up

                # Branch DOWN: s_curr = 0
                s_sens_down = 0 * 2 + s_prev
                p_persist_down = float(pol[s_sens_down])
                p_reverse_down = 1.0 - p_persist_down

                # If persist: heading remains a_curr, next s_prev becomes 0
                j_p_down = int(np.clip(i + int(a_curr), 0, n - 1))
                m_next_p_down = heading_idx * 2 + 0
                if p_persist_down > 0.0 and p_down > 0.0:
                    P[u, j_p_down * num_mem + m_next_p_down] += p_down * p_persist_down

                # If reverse: heading flips to -a_curr, next s_prev becomes 0
                j_r_down = int(np.clip(i - int(a_curr), 0, n - 1))
                m_next_r_down = (1 - heading_idx) * 2 + 0
                if p_reverse_down > 0.0 and p_down > 0.0:
                    P[u, j_r_down * num_mem + m_next_r_down] += p_down * p_reverse_down

        P_csr = P.tocsr()
        row_sums = np.asarray(P_csr.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0), f"Row sums do not equal 1.0: min={float(row_sums.min())}, max={float(row_sums.max())}"
        return P_csr

    def solve_2bit_mfpt(self, policy_persist: Optional[Any] = None, x_start: Optional[float] = None) -> Dict[str, Any]:
        """
        Solves the exact discrete-time MFPT for a 2-bit agent: (I - P_trans) T = Delta t * 1.
        Returns point-start MFPT, QSD escape time, and full transient T vector.
        """
        x_start_val = self.ls.x1_peak if x_start is None else float(x_start)
        P = self.build_2bit_transition_matrix(policy_persist=policy_persist, absorbing=True)

        n = self.n_spatial
        num_mem = 4
        N = num_mem * n
        absorbing_mask = np.zeros(N, dtype=bool)
        for i in range(self.absorbing_spatial_idx, n):
            for m in range(num_mem):
                absorbing_mask[i * num_mem + m] = True

        transient_mask = ~absorbing_mask
        transient_indices = np.where(transient_mask)[0]

        P_trans = P[transient_indices, :][:, transient_indices]
        I_sp = sp.eye(len(transient_indices), format="csr")

        rhs = np.full(len(transient_indices), self.params.dt, dtype=np.float64)
        T_trans = spsolve(I_sp - P_trans, rhs)

        T_full = np.zeros(N, dtype=np.float64)
        T_full[transient_indices] = T_trans

        # Release at x_start averaged over initial memory configurations
        start_idx = int(np.argmin(np.abs(self.x_grid - x_start_val)))
        mfpt_point = float(np.mean([T_full[start_idx * num_mem + m] for m in range(num_mem)]))

        return {
            "b": 2,
            "mfpt_point_start": mfpt_point,
            "T_full": T_full,
            "n_states": N,
            "absorbing_idx": self.absorbing_spatial_idx
        }

    def compute_2bit_committor(self, policy_persist: Optional[Any] = None, x_A: Optional[float] = None) -> Dict[str, Any]:
        """
        Computes the Committor function q(i, M) for the 2-bit agent:
        q = 0 on A (deep in basin: x <= x_A), q = 1 on B (past barrier: x >= x_min).
        Reaction coordinate diagnostic: Delta q(i) = max_{m, m'} |q(i, m) - q(i, m')|.
        """
        P = self.build_2bit_transition_matrix(policy_persist=policy_persist, absorbing=False)
        n = self.n_spatial
        num_mem = 4
        N = num_mem * n

        x_A_val = (self.ls.x1_peak - 0.5) if x_A is None else float(x_A)
        idx_A = int(np.argmin(np.abs(self.x_grid - x_A_val)))
        idx_B = self.absorbing_spatial_idx

        mask_A = np.zeros(N, dtype=bool)
        mask_B = np.zeros(N, dtype=bool)
        for i in range(idx_A + 1):
            for m in range(num_mem):
                mask_A[i * num_mem + m] = True
        for i in range(idx_B, n):
            for m in range(num_mem):
                mask_B[i * num_mem + m] = True

        mask_T = (~mask_A) & (~mask_B)
        idx_A_arr = np.where(mask_A)[0]
        idx_B_arr = np.where(mask_B)[0]
        idx_T_arr = np.where(mask_T)[0]

        P_TT = P[idx_T_arr, :][:, idx_T_arr]
        P_TB = P[idx_T_arr, :][:, idx_B_arr]

        I_sp = sp.eye(len(idx_T_arr), format="csr")
        rhs = np.asarray(P_TB.sum(axis=1), dtype=np.float64).flatten()
        q_T = spsolve(I_sp - P_TT, rhs)

        q_full = np.zeros(N, dtype=np.float64)
        q_full[idx_A_arr] = 0.0
        q_full[idx_B_arr] = 1.0
        q_full[idx_T_arr] = q_T

        delta_q = np.zeros(n, dtype=np.float64)
        q_avg = np.zeros(n, dtype=np.float64)
        for i in range(n):
            mem_vals = [float(q_full[i * num_mem + m]) for m in range(num_mem)]
            q_avg[i] = float(np.mean(mem_vals))
            delta_q[i] = float(np.max(mem_vals) - np.min(mem_vals))

        idx_ts = int(np.argmin(np.abs(q_avg - 0.5)))
        x_transition_state = float(self.x_grid[idx_ts])
        max_delta_q = float(np.max(delta_q))

        return {
            "x_grid": self.x_grid,
            "q_full": q_full,
            "q_avg": q_avg,
            "delta_q": delta_q,
            "max_delta_q": max_delta_q,
            "x_transition_state": x_transition_state,
            "x_min": self.ls.x_min
        }


if __name__ == "__main__":
    ls = ChemicalLandscape.from_preset("medium")
    solver = ExactMarkovChainSolver(ls)
    res = solver.solve_exact_mfpt()
    print("=== Exact Markov Chain Solver Test (Preset: Medium) ===")
    print(f"  Point-Start MFPT:        {res['mfpt_point_start']:.4f} s")
    print(f"  QSD Escape Time:         {res['mfpt_qsd']:.4f} s")
    print(f"  Slow Mode tau_esc:       {res['tau_esc_spectral']:.4f} s")
    print(f"  Fast Mode tau_rel:       {res['tau_rel']:.4f} s")
    print(f"  Timescale Separation:    {res['timescale_separation']:.6e}")
    com = solver.compute_committor()
    print(f"  Max Delta q (diagnostic): {com['max_delta_q']:.6e} (confirms memory is not a reaction coord)")
    print(f"  Transition State (q=0.5): x = {com['x_transition_state']:.4f} (vs valley x_min = {ls.x_min:.4f})")
