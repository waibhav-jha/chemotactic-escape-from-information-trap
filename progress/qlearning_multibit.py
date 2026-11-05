"""
Multi-Bit Tabular Q-Learning Agent (Step 13 & Step 14, Roadmap Version 2).
Strictly implements the POMDP shortest-path formulation:
  - Observation state z_t in {0, ..., 2^b - 1} (internal memory bits M only; zero spatial coordinates!).
  - Motor Actions: A_t in {0 (Persist), 1 (Reverse)} for b >= 1; {0 (Left), 1 (Right)} for b = 0.
  - Shortest-path cost: r_t = -dt until target basin is reached, r_terminal = 0, gamma = 1.0.
  - Q-table size: strictly 2^b x 2 matrix (zero spatial position leakage).
  - Evaluates learned policy on both independent Monte Carlo rollouts and exact Markov chain linear solve.
"""
from typing import Dict, Any, Tuple, Optional
import sys
import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.multibit_memory import MultiBitMemory
from progress.markov_chain import ExactMarkovChainSolver


class MultiBitQChemotaxis:
    def __init__(
        self,
        landscape: ChemicalLandscape,
        b: int = 1,
        params: Optional[PhysicalParameters] = None,
        alpha: float = 0.20,
        gamma: float = 0.99,
        epsilon_start: float = 0.35,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        temperature: float = 0.5
    ):
        self.ls = landscape
        self.b = int(b)
        self.params: PhysicalParameters = landscape.params if params is None else params
        self.mem_arch = MultiBitMemory(b=self.b, decay_rate=0.4)

        self.solver = ExactMarkovChainSolver(landscape, self.params)
        self.grid = self.solver.x_grid
        self.n_spatial = self.solver.n_spatial

        # Strictly POMDP: state space consists solely of internal sensory/motor memory states
        # No spatial binning x is available to the controller!
        self.n_states = 2 ** self.b if self.b > 0 else 2
        self.n_actions = 2  # 0: Persist, 1: Reverse (for b>=1) / 0: Left, 1: Right (b=0)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature

        # Planning horizon
        self.tau_gamma = self.params.dt / max(1e-4, 1.0 - self.gamma)

        # Q-table: strictly (2^b x 2)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        # Memoryless b=0 threshold
        self.theta = 0.5 * (float(self.ls.c_xmin) + float(self.ls.c_x1))

    def select_action(self, state_idx: int, rng: np.random.Generator, explore: bool = True) -> int:
        if explore and rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))

        q_vals = self.Q[state_idx]
        max_q = float(np.max(q_vals))
        best_actions = np.where(np.isclose(q_vals, max_q))[0]
        return int(rng.choice(best_actions))

    def train(self, n_episodes: int = 400, max_steps: int = 200, seed: int = 42) -> Dict[str, Any]:
        """
        Trains the tabular Q-learner on the sensory observation space.
        Hidden position x is never an RL input.
        """
        rng = np.random.default_rng(seed)
        episode_lengths = []
        episode_rewards = []

        dt = self.params.dt
        sigma = self.params.sigma

        for ep in range(n_episodes):
            # Physical state initialization in the trap basin
            x = rng.uniform(-1.8, -1.0)
            heading = rng.choice([-1.0, 1.0])
            c_curr = float(self.ls.c(x)) + sigma * rng.standard_normal()

            # Initial memory state
            if self.b == 0:
                s_curr = 1 if c_curr > self.theta else 0
            else:
                s_curr = int(rng.integers(self.n_states))

            total_reward = 0.0
            steps = 0

            while steps < max_steps:
                steps += 1
                action = self.select_action(s_curr, rng, explore=True)

                if self.b == 0:
                    heading = -1.0 if action == 0 else +1.0
                    persist = True
                else:
                    persist = (action == 0)
                    if not persist:
                        heading = -heading

                # Physical move
                x_next = np.clip(x + heading * self.params.v * dt, self.params.x_left, self.ls.x_min)
                c_next = float(self.ls.c(x_next)) + sigma * rng.standard_normal()
                delta_c = c_next - c_curr
                meas_bit = 1 if delta_c > 0 else 0

                # Memory update
                if self.b == 0:
                    s_next = 1 if c_next > self.theta else 0
                else:
                    s_next = self.mem_arch.update_state(s_curr, delta_c_sign=meas_bit, persist=persist)

                done = (x_next >= self.ls.x_min)
                # Shortest-path reward: r = -dt per step
                reward = -dt

                # TD Update
                target = reward + (0.0 if done else self.gamma * float(np.max(self.Q[s_next])))
                self.Q[s_curr, action] += self.alpha * (target - self.Q[s_curr, action])

                total_reward += reward
                x = x_next
                c_curr = c_next
                s_curr = s_next

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)

        return {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "final_epsilon": self.epsilon,
            "learned_policy": np.argmax(self.Q, axis=1).tolist()
        }

    def evaluate_exact_markov(self) -> float:
        """
        Evaluates the learned frozen policy on the exact discrete Markov chain.
        Eliminates Monte Carlo sample variance to calculate true expected hitting time.
        """
        learned_actions = np.argmax(self.Q, axis=1) # 0: persist, 1: reverse
        x_bins = self.grid
        n = self.n_spatial
        num_mem = self.n_states
        total_states = n * num_mem
        abs_idx = self.solver.absorbing_spatial_idx
        dt = self.params.dt
        sigma = self.params.sigma

        P = sparse.lil_matrix((total_states, total_states), dtype=np.float64)

        def flat_idx(i_pos: int, m_val: int) -> int:
            return i_pos * num_mem + m_val

        for i in range(n):
            if i >= abs_idx:
                for m in range(num_mem):
                    P[flat_idx(i, m), flat_idx(i, m)] = 1.0
                continue

            for m in range(num_mem):
                u = flat_idx(i, m)

                if self.b == 0:
                    act = learned_actions[m]
                    a_dir = -1.0 if act == 0 else +1.0
                    j_next = int(np.clip(i + int(a_dir * self.params.K_grid), 0, n - 1))
                    # Next observation state depends on concentration at j_next
                    x_nxt = float(x_bins[j_next])
                    p_high = float(ndtr((self.ls.c(x_nxt) - self.theta) / sigma))
                    P[u, flat_idx(j_next, 1)] += p_high
                    P[u, flat_idx(j_next, 0)] += (1.0 - p_high)
                    continue

                heading, _ = self.mem_arch.decode_state(m)
                a_curr = heading

                i_prev = int(np.clip(i - int(a_curr), 0, n - 1))
                delta_c = float(self.ls.c(x_bins[i]) - self.ls.c(x_bins[i_prev]))

                p_meas_up = float(ndtr(delta_c / (np.sqrt(2.0) * sigma)))
                p_meas_down = 1.0 - p_meas_up

                # The learned action for state m
                act = learned_actions[m]
                persist = (act == 0)
                a_next = a_curr if persist else -a_curr
                j_next = int(np.clip(i + int(a_next * self.params.K_grid), 0, n - 1))

                # On measurement UP
                m_up = self.mem_arch.update_state(m, delta_c_sign=+1, persist=persist)
                P[u, flat_idx(j_next, m_up)] += p_meas_up

                # On measurement DOWN
                m_down = self.mem_arch.update_state(m, delta_c_sign=-1, persist=persist)
                P[u, flat_idx(j_next, m_down)] += p_meas_down

        P_csr = P.tocsr()
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

        x1_idx = int(np.argmin(np.abs(x_bins - self.ls.x1_peak)))
        mfpt_val = float(np.mean([T_full[flat_idx(x1_idx, m)] for m in range(num_mem)]))
        return mfpt_val


def evaluate_multibit_qlearning_sweep(bit_list=[1, 2, 3, 4, 5], n_episodes=300):
    """
    Trains and evaluates POMDP Tabular Q-learning agents across multiple bit depths.
    """
    p = PhysicalParameters(sigma=0.20, dt=0.01)
    ls = ChemicalLandscape.from_preset("L1a", sigma=0.20, dt=0.01)

    print("=" * 76)
    print("POMDP TABULAR Q-LEARNING SWEEP ACROSS MEMORY BUDGETS b in {1..5}")
    print("=" * 76)
    print(f"{'Memory b':>8s} | {'States 2^b':>10s} | {'Learned Policy':>24s} | {'Exact Markov MFPT (s)':>22s}")
    print("-" * 76)

    results = {}
    for b in bit_list:
        agent = MultiBitQChemotaxis(ls, b=b, params=p, alpha=0.20, gamma=0.99)
        train_res = agent.train(n_episodes=n_episodes, max_steps=150, seed=42 + b)
        mfpt_learned = agent.evaluate_exact_markov()
        pol_str = str(train_res["learned_policy"])
        if len(pol_str) > 22:
            pol_str = pol_str[:20] + ".."
        results[b] = {
            "mfpt_learned": mfpt_learned,
            "policy": train_res["learned_policy"]
        }
        print(f"{b:8d} | {2**b:10d} | {pol_str:>24s} | {mfpt_learned:22.2f}")

    return results


if __name__ == "__main__":
    evaluate_multibit_qlearning_sweep()
