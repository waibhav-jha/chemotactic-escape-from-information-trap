"""
Information-Theoretic & Non-Equilibrium Thermodynamic Diagnostics (Step 9+).
Includes:
1. Mutual Information I(a_t; c_{t-k}) across temporal lags k
2. Effective Memory Horizon k_eff
3. Berg-Purcell Physical Sensing SNR Limit & Memory Saturation
4. Landauer Erasure Cost & Rate-Distortion Pareto Frontier
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters, THERMODYNAMIC_PARAMS


def compute_mutual_information_lags(
    actions: np.ndarray,
    measurements: np.ndarray,
    max_lag: int = 10
) -> Dict[str, np.ndarray]:
    """
    Computes Mutual Information I(a_t; s_{t-k}) between action at time t
    and binary sensory difference s_{t-k} = sign(c_{t-k} - c_{t-k-1}) for lags k=1..max_lag.

    MI is expressed in bits:
        I(X; Y) = sum_{x,y} p(x, y) * log2( p(x, y) / (p(x) * p(y)) )
    """
    N = len(actions)
    lags = np.arange(1, max_lag + 1)
    mi_vals = np.zeros(len(lags), dtype=np.float64)

    # Convert actions to binary {0, 1}
    a_binary = (actions > 0).astype(np.int32)

    # Sensory differences
    diffs = np.diff(measurements)
    s_binary = (diffs > 0).astype(np.int32)
    # Align length
    a_aligned = a_binary[1:]
    L = len(s_binary)

    for idx, k in enumerate(lags):
        if k >= L - 10:
            continue
        # Paired sequence: a[t] with s[t-k]
        a_k = a_aligned[k:]
        s_k = s_binary[:-k] if k > 0 else s_binary

        min_len = min(len(a_k), len(s_k))
        a_k = a_k[:min_len]
        s_k = s_k[:min_len]

        # 2x2 contingency table
        counts = np.zeros((2, 2), dtype=np.float64)
        for val_a, val_s in zip(a_k, s_k):
            counts[val_a, val_s] += 1.0

        p_joint = counts / np.sum(counts)
        p_a = np.sum(p_joint, axis=1)
        p_s = np.sum(p_joint, axis=0)

        # MI calculation with safe log
        mi = 0.0
        for i in range(2):
            for j in range(2):
                if p_joint[i, j] > 1e-12 and p_a[i] > 1e-12 and p_s[j] > 1e-12:
                    mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_a[i] * p_s[j]))
        mi_vals[idx] = max(0.0, mi)

    return {
        "lags": lags,
        "mutual_information": mi_vals,
        "effective_memory_horizon": int(lags[np.argmax(mi_vals < 0.05)]) if np.any(mi_vals < 0.05) else int(max_lag)
    }


def compute_berg_purcell_limit(
    gradient: float,
    velocity: float,
    sensing_noise: float,
    dt: float,
    memory_bits: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes theoretical Berg-Purcell SNR scaling and predicted saturation:
        SNR_BP ~ (g * v * tau^(3/2)) / sigma
        tau = m * dt, m = 2^b (or m = b for shift register)
    """
    tau_vals = np.maximum(1, memory_bits) * dt
    snr = (np.abs(gradient) * velocity * (tau_vals ** 1.5)) / max(1e-4, sensing_noise)

    # Theoretical optimal integration window balancing noise reduction vs spatial staleness
    # tau* ~ (sigma / (g * v))^(2/3)
    tau_star = (sensing_noise / max(1e-4, np.abs(gradient) * velocity)) ** (2.0 / 3.0)
    b_star_theoretical = max(1.0, tau_star / dt)

    # Information saturation factor (0 to 1)
    fidelity = 1.0 - np.exp(-snr / np.maximum(1.0, np.max(snr) * 0.5))

    return {
        "memory_bits": memory_bits,
        "tau_vals": tau_vals,
        "snr_bp": snr,
        "tau_star": float(tau_star),
        "b_star_theoretical": float(b_star_theoretical),
        "fidelity": fidelity
    }


def compute_landauer_pareto_frontier(
    memory_bits: np.ndarray,
    mfpt_values: np.ndarray,
    beta_weights: Optional[List[float]] = None
) -> Dict[str, object]:
    """
    Computes:
    1. Landauer thermodynamic dissipation cost: C_erasure = b * kB*T * ln(2)
    2. Net cost curve: L(b, beta) = <T_esc>(b) + beta * b * ln(2)
    3. Rate-Distortion Pareto frontier: Escape Rate J*(R) = 1 / <T_esc>*(R)
    """
    if beta_weights is None:
        beta_weights = THERMODYNAMIC_PARAMS["beta_sweep"]

    ln2 = THERMODYNAMIC_PARAMS["ln2"]
    kB_T = THERMODYNAMIC_PARAMS["kB_T"]

    # Erasure energy cost per update
    erasure_costs = memory_bits * kB_T * ln2

    # Escape rate J = 1 / MFPT
    escape_rates = 1.0 / np.maximum(1e-5, mfpt_values)

    # Net cost curves for each beta
    net_costs = {}
    optimal_b_per_beta = {}
    for beta in beta_weights:
        cost = mfpt_values + beta * erasure_costs
        net_costs[f"beta_{beta}"] = cost
        opt_idx = int(np.argmin(cost))
        optimal_b_per_beta[float(beta)] = int(memory_bits[opt_idx])

    # Pareto frontier points: convex/concave non-dominated trade-off between R (bits) and J (escape rate)
    pareto_bits = []
    pareto_rates = []
    max_rate_so_far = -1.0
    for b_val, rate in zip(memory_bits, escape_rates):
        if rate > max_rate_so_far:
            pareto_bits.append(int(b_val))
            pareto_rates.append(float(rate))
            max_rate_so_far = rate

    return {
        "memory_bits": memory_bits,
        "mfpt_values": mfpt_values,
        "escape_rates": escape_rates,
        "erasure_costs": erasure_costs,
        "net_costs": net_costs,
        "optimal_b_per_beta": optimal_b_per_beta,
        "pareto_bits": np.array(pareto_bits),
        "pareto_rates": np.array(pareto_rates)
    }
