"""
10_qlearning.py: Tabular Q-learning agent on (x_bin, M_t) with true-field scalar reward r ~ c(x).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.landscape import ChemicalLandscape
from progress.qlearning import TabularQChemotaxis

if __name__ == "__main__":
    print("=== [10_qlearning.py] Tabular Q-Learning on Information Traps (b = 1, 2) ===")
    ls = ChemicalLandscape.from_preset("shallow")

    # 1. 1-Bit Agent
    print("\n--- 1-Bit Q-Learning Agent ---")
    agent1 = TabularQChemotaxis(ls, b=1, alpha=0.20, gamma=0.98)
    train_res1 = agent1.train(n_episodes=500)
    gap1 = agent1.compute_learning_gap()
    print(f"Training Complete ({len(train_res1['episode_rewards'])} episodes)")
    print(f"Learned Frozen Policy MFPT:    {gap1['mfpt_learned']:10.3f} s")
    print(f"Optimal Ground Truth MFPT:     {gap1['mfpt_optimal']:10.3f} s")
    print(f"Quantitative Learning Gap:     {gap1['learning_gap']:10.3f} ({gap1['ratio_percentage']})")
    print(f"Committor Delta q:             {gap1['max_delta_q']:.6e} (Memory non-reactive)")

    # 2. 2-Bit Agent
    print("\n--- 2-Bit Q-Learning Agent (Temporal Filter) ---")
    agent2 = TabularQChemotaxis(ls, b=2, alpha=0.20, gamma=0.98)
    train_res2 = agent2.train(n_episodes=500)
    gap2 = agent2.compute_learning_gap()
    print(f"Training Complete ({len(train_res2['episode_rewards'])} episodes)")
    print(f"Learned Frozen Policy MFPT:    {gap2['mfpt_learned']:10.3f} s")
    print(f"Optimal Ground Truth MFPT:     {gap2['mfpt_optimal']:10.3f} s")
    print(f"Quantitative Learning Gap:     {gap2['learning_gap']:10.3f} ({gap2['ratio_percentage']})")
    print(f"Committor Delta q:             {gap2['max_delta_q']:.6e} (Memory ENTERS reaction coordinate!)")
