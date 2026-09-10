"""
Multi-Bit Tabular Q-Learning Agent (Step 9+).
Scales to b in {0, 1, 2, 3, 4, 5, 6, 8, 10} bits.
Features:
- Shift-register memory state M_t in {0, ..., 2^b - 1}
- Boltzmann softmax exploration alongside epsilon-greedy
- Physical planning horizon tau_gamma = dt / (1 - gamma)
- Frozen-policy evaluation (<T_esc>, trajectories, action/measurement logs)
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters
from progress.landscape import ChemicalLandscape
from progress.multibit_memory import MultiBitMemory


class MultiBitQChemotaxis:
    def __init__(
        self,
        landscape: ChemicalLandscape,
        b: int = 1,
        params: Optional[PhysicalParameters] = None,
        alpha: float = 0.20,
        gamma: float = 0.992,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.996,
        temperature: float = 0.5
    ):
        self.ls = landscape
        self.b = int(b)
        self.params = landscape.params if params is None else params
        self.mem_arch = MultiBitMemory(b=self.b, decay_rate=0.4)

        from progress.markov_chain import ExactMarkovChainSolver
        self.solver = ExactMarkovChainSolver(landscape, self.params)
        self.grid = self.solver.x_grid
        self.n_spatial = self.solver.n_spatial
        self.n_memory = 2 ** self.b if self.b > 0 else 1
        self.n_states = self.n_spatial * self.n_memory
        self.n_actions = 2  # 0: move left (-1), 1: move right (+1)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature

        # Physical planning horizon
        self.tau_gamma = self.params.dt / (1.0 - self.gamma)

        # Q-table: states x actions
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def state_to_idx(self, bin_idx: int, mem_idx: int) -> int:
        return bin_idx * self.n_memory + mem_idx

    def idx_to_state(self, state_idx: int) -> Tuple[int, int]:
        return state_idx // self.n_memory, state_idx % self.n_memory

    def select_action(self, state_idx: int, rng: np.random.Generator, explore: bool = True, boltzmann: bool = False) -> int:
        if explore:
            if boltzmann:
                # Softmax action selection
                q = self.Q[state_idx] / max(1e-3, self.temperature)
                exp_q = np.exp(q - np.max(q))
                probs = exp_q / np.sum(exp_q)
                return int(rng.choice(self.n_actions, p=probs))
            elif rng.random() < self.epsilon:
                return int(rng.integers(self.n_actions))

        # Greedy choice
        q_vals = self.Q[state_idx]
        max_q = float(np.max(q_vals))
        best = np.where(np.isclose(q_vals, max_q))[0]
        return int(rng.choice(best))

    def train(self, n_episodes: int = 400, max_steps: int = 1500, seed: int = 42) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        episode_lengths = []
        episode_rewards = []

        abs_spatial_idx = int(np.argmin(np.abs(self.grid - (self.ls.x_min + self.params.absorbing_delta))))
        start_spatial_idx = int(np.argmin(np.abs(self.grid - self.ls.x1_peak)))

        v = self.params.v
        dt = self.params.dt
        sigma = self.params.sigma

        for ep in range(n_episodes):
            bin_idx = start_spatial_idx
            mem_idx = int(rng.integers(self.n_memory)) if self.b > 0 else 0
            state_idx = self.state_to_idx(bin_idx, mem_idx)

            # Initial sensor measurement
            x_curr = float(self.grid[bin_idx])
            c_curr = float(self.ls.c(x_curr)) + sigma * rng.standard_normal()

            total_reward = 0.0
            steps = 0

            while steps < max_steps and bin_idx < abs_spatial_idx:
                steps += 1
                action_idx = self.select_action(state_idx, rng, explore=True, boltzmann=(ep % 2 == 1))
                action_dir = -1 if action_idx == 0 else +1

                # Spatial displacement
                next_bin = int(np.clip(bin_idx + action_dir * self.params.K_grid, 0, self.n_spatial - 1))
                x_next = float(self.grid[next_bin])

                # Noisy sensor measurement
                c_next = float(self.ls.c(x_next)) + sigma * rng.standard_normal()
                delta_c = c_next - c_curr
                meas_bit = 1 if delta_c > 0 else 0

                # Memory update
                if self.b > 0:
                    persist = (action_dir > 0) == ((mem_idx & 1) == 1)
                    next_mem = int(self.mem_arch.update_state(mem_idx, delta_c_sign=meas_bit, persist=persist))
                else:
                    next_mem = 0

                next_state_idx = self.state_to_idx(next_bin, next_mem)

                done = (next_bin >= abs_spatial_idx)
                # True field reward + barrier achievement reward
                reward = float(self.ls.c(x_next))
                if done:
                    reward += 100.0

                # Q-learning TD update
                target = reward
                if not done:
                    target += self.gamma * float(np.max(self.Q[next_state_idx]))
                self.Q[state_idx, action_idx] += self.alpha * (target - self.Q[state_idx, action_idx])

                total_reward += reward
                c_curr = c_next
                bin_idx = next_bin
                mem_idx = next_mem
                state_idx = next_state_idx

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)

        return {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "final_epsilon": self.epsilon
        }

    def evaluate_frozen_policy(
        self,
        n_eval_trajectories: int = 500,
        max_eval_steps: int = 15000,
        seed: int = 123
    ) -> Dict[str, Any]:
        """
        Evaluates the trained policy with exploration completely disabled (epsilon = 0).
        Collects trajectory histories for mutual information calculation.
        """
        rng = np.random.default_rng(seed)
        abs_spatial_idx = int(np.argmin(np.abs(self.grid - (self.ls.x_min + self.params.absorbing_delta))))
        start_spatial_idx = int(np.argmin(np.abs(self.grid - self.ls.x1_peak)))

        escape_times = []
        action_history = []
        meas_history = []

        dt = self.params.dt
        sigma = self.params.sigma

        for _ in range(n_eval_trajectories):
            bin_idx = start_spatial_idx
            mem_idx = 0
            state_idx = self.state_to_idx(bin_idx, mem_idx)

            x_curr = float(self.grid[bin_idx])
            c_curr = float(self.ls.c(x_curr)) + sigma * rng.standard_normal()

            step = 0
            while step < max_eval_steps and bin_idx < abs_spatial_idx:
                step += 1
                action_idx = self.select_action(state_idx, rng, explore=False)
                action_dir = -1 if action_idx == 0 else +1

                if len(action_history) < 20000:
                    action_history.append(action_dir)
                    meas_history.append(c_curr)

                next_bin = int(np.clip(bin_idx + action_dir * self.params.K_grid, 0, self.n_spatial - 1))
                x_next = float(self.grid[next_bin])
                c_next = float(self.ls.c(x_next)) + sigma * rng.standard_normal()
                delta_c = c_next - c_curr
                meas_bit = 1 if delta_c > 0 else 0

                if self.b > 0:
                    persist = (action_dir > 0) == ((mem_idx & 1) == 1)
                    next_mem = int(self.mem_arch.update_state(mem_idx, delta_c_sign=meas_bit, persist=persist))
                else:
                    next_mem = 0

                bin_idx = next_bin
                mem_idx = next_mem
                state_idx = self.state_to_idx(bin_idx, mem_idx)
                c_curr = c_next

            escape_times.append(step * dt)

        mean_mfpt = float(np.mean(escape_times))
        std_mfpt = float(np.std(escape_times))

        return {
            "mean_mfpt": mean_mfpt,
            "std_mfpt": std_mfpt,
            "escape_times": escape_times,
            "actions": np.array(action_history, dtype=np.int32),
            "measurements": np.array(meas_history, dtype=np.float64)
        }
