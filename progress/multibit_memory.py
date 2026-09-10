"""
Multi-bit Memory Architectures and Transition Dynamics (Step 9+).
Physically Matched Representation:
- b = 0: Memoryless (0 bits).
- b = 1: 1 motor bit (heading direction a in {-1, +1}). Run-and-tumble persist-on-rise, reverse-on-fall.
- b >= 2: 1 motor bit (heading a) + (b-1) sensory history bits (temporal gradient filter).
Exact sparse Markov transition builder on joint state (x_bin, M) for b in {0, 1, 2, 3}.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr
from typing import Tuple, Dict, Optional, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.markov_chain import ExactMarkovChainSolver


class MultiBitMemory:
    """
    Handles internal state representations and transitions for an agent
    with b bits of internal memory.
    Internal state integer M in [0, 2^b - 1]:
      - If b == 0: M = 0
      - If b >= 1:
          dir_idx = M & 1 (0 for -1.0, 1 for +1.0)
          sensory_bits = M >> 1 ((b-1) bits of past gradient signs)
    """
    def __init__(self, b: int, decay_rate: float = 0.4):
        self.b = int(b)
        self.num_memory_states = 2 ** self.b if self.b > 0 else 1
        self.decay_rate = decay_rate

        # Precompute normalized filter weights for sensory history (length b-1)
        if self.b <= 1:
            self.weights = np.array([1.0])
        else:
            sensory_len = self.b
            lags = np.arange(sensory_len)
            w = np.exp(-self.decay_rate * lags)
            self.weights = w / np.sum(w)

    def decode_state(self, M: int) -> Tuple[float, np.ndarray]:
        """
        Decodes integer M into (current_heading, past_sensory_bits).
        heading in {-1.0, +1.0}
        past_sensory_bits in {-1.0, +1.0} of length max(1, b-1)
        """
        if self.b == 0:
            return 1.0, np.array([1.0])
        dir_idx = M & 1
        heading = +1.0 if dir_idx == 1 else -1.0

        if self.b == 1:
            return heading, np.array([1.0])

        n_sensory = self.b - 1
        bits = np.zeros(n_sensory, dtype=np.float64)
        sensory_int = M >> 1
        for k in range(n_sensory):
            val = (sensory_int >> k) & 1
            bits[k] = +1.0 if val == 1 else -1.0
        return heading, bits

    def update_state(self, M: int, delta_c_sign: int, persist: bool) -> int:
        """
        Updates memory state M given current measurement sign in {0, 1} and persist decision.
        """
        if self.b == 0:
            return 0

        curr_dir = M & 1
        next_dir = curr_dir if persist else (1 - curr_dir)

        if self.b == 1:
            return next_dir

        n_sensory = self.b - 1
        sensory_int = M >> 1
        next_sensory = ((sensory_int << 1) | (1 if delta_c_sign > 0 else 0)) & ((1 << n_sensory) - 1)
        return (next_sensory << 1) | next_dir

    def decide_persist(self, M: int, current_meas_sign: float) -> bool:
        """
        Bayesian temporal filter:
        Combines current measurement sign with historical sensory bits.
        Returns True to persist, False to reverse.
        """
        if self.b == 0:
            return True
        if self.b == 1:
            return current_meas_sign > 0.0

        heading, past_bits = self.decode_state(M)
        # All evidence vector: current reading at index 0, past bits at index 1..
        all_bits = np.concatenate(([current_meas_sign], past_bits))
        evidence = np.dot(all_bits[:len(self.weights)], self.weights)
        return evidence >= 0.0


def build_multibit_markov_chain(
    landscape: ChemicalLandscape,
    params: PhysicalParameters,
    b: int,
    filter_decay: float = 0.4
) -> Tuple[sparse.csr_matrix, np.ndarray, float]:
    """
    Constructs the exact discrete Markov transition matrix on joint state (x_bin, M)
    for b in {0, 1, 2, 3}.
    For b = 1, this matches ExactMarkovChainSolver identically.
    For b = 2, 3, it evaluates the multi-bit filtered policy.
    """
    solver = ExactMarkovChainSolver(landscape, params)
    x_bins = solver.x_grid
    n = solver.n_spatial
    num_mem = 2 ** b if b > 0 else 1
    total_states = n * num_mem

    mem_arch = MultiBitMemory(b=b, decay_rate=filter_decay)
    abs_idx = solver.absorbing_spatial_idx
    sigma = params.sigma
    dt = params.dt

    P = sparse.lil_matrix((total_states, total_states), dtype=np.float64)

    def flat_idx(i_pos: int, m_val: int) -> int:
        return i_pos * num_mem + m_val

    for i in range(n):
        if i >= abs_idx:
            # Absorbing boundary
            for m in range(num_mem):
                u = flat_idx(i, m)
                P[u, u] = 1.0
            continue

        for m in range(num_mem):
            u = flat_idx(i, m)

            if b == 0:
                # Memoryless agent: chooses left or right based on local concentration threshold
                # Symmetrically wanders
                j_left = max(0, i - params.K_grid)
                j_right = min(n - 1, i + params.K_grid)
                P[u, flat_idx(j_left, 0)] += 0.5
                P[u, flat_idx(j_right, 0)] += 0.5
                continue

            heading, _ = mem_arch.decode_state(m)
            a_curr = heading  # -1.0 or +1.0

            # The agent arrived at bin i having moved with heading a_curr
            i_prev = int(np.clip(i - int(a_curr), 0, n - 1))
            x_curr = float(x_bins[i])
            x_prev = float(x_bins[i_prev])

            delta_c = float(landscape.c(x_curr) - landscape.c(x_prev))

            # Sensing probabilities
            p_meas_up = float(ndtr(delta_c / (np.sqrt(2.0) * sigma)))
            p_meas_down = 1.0 - p_meas_up

            # 1. Branch where current measurement is UP (+1)
            persist_on_up = mem_arch.decide_persist(m, current_meas_sign=+1.0)
            a_on_up = a_curr if persist_on_up else -a_curr
            j_up = int(np.clip(i + int(a_on_up), 0, n - 1))
            m_next_up = mem_arch.update_state(m, delta_c_sign=+1, persist=persist_on_up)
            P[u, flat_idx(j_up, m_next_up)] += p_meas_up

            # 2. Branch where current measurement is DOWN (-1)
            persist_on_down = mem_arch.decide_persist(m, current_meas_sign=-1.0)
            a_on_down = a_curr if persist_on_down else -a_curr
            j_down = int(np.clip(i + int(a_on_down), 0, n - 1))
            m_next_down = mem_arch.update_state(m, delta_c_sign=-1, persist=persist_on_down)
            P[u, flat_idx(j_down, m_next_down)] += p_meas_down

    P_csr = P.tocsr()
    # Row sum assert
    row_sums = np.asarray(P_csr.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 1.0), f"Row sums != 1.0: min={row_sums.min()}, max={row_sums.max()}"

    # Extract transient submatrix
    absorbing_indices = [flat_idx(abs_idx, m) for m in range(num_mem)]
    transient_mask = np.ones(total_states, dtype=bool)
    transient_mask[absorbing_indices] = False
    transient_indices = np.where(transient_mask)[0]

    P_trans = P_csr[transient_indices, :][:, transient_indices]
    I_sp = sparse.eye(len(transient_indices), format="csr")

    rhs = np.full(len(transient_indices), dt, dtype=np.float64)
    T_trans = spsolve(I_sp - P_trans, rhs)

    T_full = np.zeros(total_states, dtype=np.float64)
    T_full[transient_indices] = T_trans

    x1_idx = int(np.argmin(np.abs(x_bins - landscape.x1_peak)))
    mfpt_x1 = float(np.mean([T_full[flat_idx(x1_idx, m)] for m in range(num_mem)]))

    return P_trans, T_full, mfpt_x1
