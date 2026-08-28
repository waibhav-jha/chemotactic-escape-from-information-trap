"""
10_qlearning.py: Tabular Q-learning agent on (x_bin, M_t) with true-field scalar reward r ~ c(x).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.landscape import ChemicalLandscape
from progress.qlearning import TabularQChemotaxis

if __name__ == "__main__":
    print("=== [10_qlearning.py] Tabular Q-Learning on Information Traps ===")
    ls = ChemicalLandscape.from_preset("shallow")
    agent = TabularQChemotaxis(ls, b=1, alpha=0.20, gamma=0.98)
    train_res = agent.train(n_episodes=500)
    gap_res = agent.compute_learning_gap()

    print(f"Training Complete ({len(train_res['episode_rewards'])} episodes)")
    print(f"Learned Frozen Policy MFPT:  {gap_res['mfpt_learned']:10.3f} s")
    print(f"Optimal Ground Truth MFPT:   {gap_res['mfpt_optimal']:10.3f} s")
    print(f"The Quantitative Learning Gap: {gap_res['learning_gap']:10.3f} ({gap_res['ratio_percentage']})")
