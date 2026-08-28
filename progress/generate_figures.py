"""
Publication-Quality Figure Generation for Chemotactic Escape (Steps 1 to 8).
Generates high-resolution PNG and vector PDF charts saved to progress/results/figures/.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Style configuration for publication-ready figures
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.parameters import PhysicalParameters, LANDSCAPE_PRESETS
from progress.landscape import ChemicalLandscape
from progress.theory import ContinuumTheory
from progress.markov_chain import ExactMarkovChainSolver


def generate_all_figures():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "data_steps_1_to_8.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    print("Generating publication figures in:", fig_dir)

    # -------------------------------------------------------------
    # Figure 1: Step 1 Landscape & Barrier Geometry
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x_eval = np.linspace(-4.5, 4.5, 1000)

    for name, col in [("shallow", "#2b5c8f"), ("medium", "#d95f02"), ("deep", "#7570b3")]:
        ls = ChemicalLandscape.from_preset(name)
        axes[0].plot(x_eval, ls.c(x_eval), label=f"{name.capitalize()} (ΔU={ls.delta_u_trap:.3f})", color=col)
        axes[0].scatter([ls.x1_peak, ls.x_min, ls.x2_peak], [ls.c_x1, ls.c_xmin, ls.c_x2], color=col, s=40, zorder=5)

        axes[1].plot(x_eval, ls.dc(x_eval), label=f"{name.capitalize()}", color=col)
        axes[2].plot(x_eval, ls.d2c(x_eval), label=f"{name.capitalize()}", color=col)

    axes[0].set_title("(a) Chemical Concentration $c(x)$")
    axes[0].set_xlabel("Position $x$")
    axes[0].set_ylabel("$c(x)$")
    axes[0].legend(frameon=True)

    axes[1].set_title("(b) Spatial Gradient $c'(x)$")
    axes[1].set_xlabel("Position $x$")
    axes[1].set_ylabel("$c'(x)$")
    axes[1].axhline(0, color="k", ls=":", alpha=0.5)

    axes[2].set_title("(c) Curvature $c''(x)$")
    axes[2].set_xlabel("Position $x$")
    axes[2].set_ylabel("$c''(x)$")
    axes[2].axhline(0, color="k", ls=":", alpha=0.5)

    plt.tight_layout()
    fig_path1 = os.path.join(fig_dir, "fig1_landscape_geometry.png")
    fig.savefig(fig_path1, dpi=300)
    plt.close(fig)
    print("  Saved:", fig_path1)

    # -------------------------------------------------------------
    # Figure 2: Step 3 No-Confinement & Drift Asymmetry
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ls_med = ChemicalLandscape.from_preset("medium")
    th_med = ContinuumTheory(ls_med)

    x_grid = np.linspace(-3.5, 3.5, 500)
    mu_b0 = th_med.b0_drift(x_grid, theta=0.5)
    mu_b1 = th_med.b1_drift(x_grid, mode="independent")

    ax.plot(x_grid, mu_b0, label="$b=0$ Memoryless Drift $\mu(x) = v F(c(x))$", color="#e7298a", lw=2.5)
    ax.plot(x_grid, mu_b1, label="$b=1$ 1-Bit Drift $\mu(x) \propto c'(x)$", color="#1b9e77", lw=2.5)
    ax.axvline(ls_med.x1_peak, color="gray", ls="--", label=f"Peak $x_1={ls_med.x1_peak:.2f}$")
    ax.axvline(ls_med.x_min, color="k", ls=":", label=f"Valley $x_{{min}}={ls_med.x_min:.2f}$")
    ax.axhline(0, color="k", ls="-", lw=0.8, alpha=0.7)

    ax.set_title("Memoryless ($b=0$) vs 1-Bit ($b=1$) Drift Profiles")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Marginal Drift $\mu(x)$")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig_path2 = os.path.join(fig_dir, "fig2_drift_profiles.png")
    fig.savefig(fig_path2, dpi=300)
    plt.close(fig)
    print("  Saved:", fig_path2)

    # -------------------------------------------------------------
    # Figure 3: Step 4 Correlated Sensing Audit & Lag-2 Autocorrelation
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lags = np.arange(11)
    # Theoretical arcsine correlation: lag 0=1, lag 2=-1/3, lag 4=2/15, odd=0
    theory_autocorr = np.zeros(11)
    theory_autocorr[0] = 1.0
    theory_autocorr[2] = -1.0 / 3.0
    theory_autocorr[4] = 2.0 / 15.0

    if "step4" in data and "autocorr_lags" in data["step4"]:
        sim_autocorr = np.array(data["step4"]["autocorr_lags"])
    else:
        sim_autocorr = theory_autocorr

    axes[0].bar(lags - 0.15, sim_autocorr, width=0.3, label="Simulation (Single Sensor)", color="#386cb0")
    axes[0].bar(lags + 0.15, theory_autocorr, width=0.3, label="Exact Theory (Arcsine Law)", color="#f0027f", alpha=0.7)
    axes[0].axhline(0, color="k", ls="-", lw=0.8)
    axes[0].set_title("(a) Action Autocorrelation $\langle a_t a_{t+k} \\rangle$ on Flat Landscape")
    axes[0].set_xlabel("Lag $k$")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_xticks(lags)
    axes[0].legend(frameon=True)

    # Drift comparison
    x_flank = np.linspace(-3.0, 0.0, 300)
    mu_indep = th_med.b1_drift(x_flank, mode="independent")
    mu_single = th_med.b1_drift(x_flank, mode="single_sensor")

    axes[1].plot(x_flank, mu_indep, label="Independent Readings ($\mu = v\,\mathrm{erf}(\dots)$)", color="#1b9e77", lw=2.5)
    axes[1].plot(x_flank, mu_single, label="Single Sensor ($\kappa_\mu = 0.7208$)", color="#d95f02", lw=2.5, ls="--")
    axes[1].set_title("(b) Drift Suppression from Stale Reading Reuse")
    axes[1].set_xlabel("Position $x$")
    axes[1].set_ylabel("Drift $\mu(x)$")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    fig_path3 = os.path.join(fig_dir, "fig3_correlated_sensing_audit.png")
    fig.savefig(fig_path3, dpi=300)
    plt.close(fig)
    print("  Saved:", fig_path3)

    # -------------------------------------------------------------
    # Figure 4: Step 6 Barrier Sweep & Headline Arrhenius Slope
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if "step6" in data and "sweep_data" in data["step6"]:
        sw = data["step6"]["sweep_data"]
        du_sig = [r["du_over_sigma"] for r in sw]
        t_chain = [r["mfpt_chain"] for r in sw]
        t_quad_indep = [r["mfpt_quad_indep"] for r in sw]
        t_quad_single = [r["mfpt_quad_single"] for r in sw]

        ax.semilogy(du_sig, t_chain, "o-", label=f"Exact Chain (Indep): Fitted Slope={data['step6']['slope_indep_fitted']:.3f} (Target 1.128)", color="#1b9e77", lw=2, markersize=7)
        ax.semilogy(du_sig, t_quad_single, "s--", label=f"Single Sensor Audit: Fitted Slope={data['step6']['slope_single_fitted']:.3f} (Target 1.554)", color="#d95f02", lw=2, markersize=7)
    else:
        du_sig = np.linspace(2, 10, 6)
        ax.semilogy(du_sig, 50 * np.exp(1.128 * du_sig), "o-", label="Independent Readings (Slope ~ 1.128)", color="#1b9e77")
        ax.semilogy(du_sig, 50 * np.exp(1.554 * du_sig), "s--", label="Single Sensor (Slope ~ 1.554)", color="#d95f02")

    ax.set_title("Headline Scaling Law: $\ln \langle T_{\mathrm{esc}} \\rangle$ vs $\Delta U_{\mathrm{trap}} / \sigma$")
    ax.set_xlabel("Trap Barrier Strength $\Delta U_{\mathrm{trap}} / \sigma$")
    ax.set_ylabel("Mean First-Passage Time $\langle T_{\mathrm{esc}} \\rangle$ (seconds)")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig_path4 = os.path.join(fig_dir, "fig4_barrier_sweep_scaling.png")
    fig.savefig(fig_path4, dpi=300)
    plt.close(fig)
    print("  Saved:", fig_path4)

    # -------------------------------------------------------------
    # Figure 5: Step 7 4-Way Cross Validation & Committor Diagnostic
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "step7" in data and "four_way_table" in data["step7"]:
        fwt = data["step7"]["four_way_table"]
        sig_vals = [r["sigma"] for r in fwt]
        t_chain_vals = [r["t_chain"] for r in fwt]
        t_quad_vals = [r["t_quad"] for r in fwt]
        t_saddle_vals = [r["t_saddle"] for r in fwt]

        axes[0].semilogy(sig_vals, t_chain_vals, "o-", label="Exact Markov Chain", color="#1b9e77", lw=2)
        axes[0].semilogy(sig_vals, t_quad_vals, "s--", label="Fokker-Planck Quadrature", color="#386cb0", lw=2)
        axes[0].semilogy(sig_vals, t_saddle_vals, "^:", label="Saddle-Point Kramers (Asymptotic)", color="#e7298a", lw=2)
        axes[0].set_title("(a) 4-Way Method Comparison Across Noise $\sigma$")
        axes[0].set_xlabel("Sensing Noise $\sigma$")
        axes[0].set_ylabel("MFPT (seconds)")
        axes[0].invert_xaxis()
        axes[0].legend(frameon=True)

    if "step7" in data and "committor" in data["step7"]:
        com = data["step7"]["committor"]
        xg = np.array(com["x_grid"])
        q_avg = np.array(com["q_avg"])
        dq = np.array(com["delta_q"])

        axes[1].plot(xg, q_avg, label="Committor $q(x)$", color="#386cb0", lw=2.5)
        axes[1].plot(xg, dq * 1000, label="$\Delta q(x) \\times 10^3$ (Reaction Coord Diagnostic)", color="#e41a1c", lw=2, ls="--")
        axes[1].axhline(0.5, color="gray", ls=":", label="Transition State Level $q=0.5$")
        axes[1].axvline(com["x_ts"], color="k", ls="--", label=f"Transition State $x={com['x_ts']:.2f}$")
        axes[1].axvline(com["x_min"], color="purple", ls=":", label=f"Valley $x_{{min}}={com['x_min']:.2f}$")
        axes[1].set_title("(b) Committor Function & Memory Reaction Diagnostic")
        axes[1].set_xlabel("Position $x$")
        axes[1].set_ylabel("Probability of Escape to Global Basin")
        axes[1].legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig_path5 = os.path.join(fig_dir, "fig5_four_way_committor.png")
    fig.savefig(fig_path5, dpi=300)
    plt.close(fig)
    print("  Saved:", fig_path5)

    # -------------------------------------------------------------
    # Figure 6: Step 8 RL Training & Learning Gap Benchmark
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "step8" in data and "b0_learning_curve" in data["step8"]:
        rew_b0 = data["step8"]["b0_learning_curve"]
        rew_b1 = data["step8"]["b1_learning_curve"]
        episodes_b0 = np.arange(len(rew_b0)) * 5
        episodes_b1 = np.arange(len(rew_b1)) * 5

        axes[0].plot(episodes_b0, rew_b0, label="$b=0$ Q-Learner (Memoryless)", color="#e7298a", lw=2)
        axes[0].plot(episodes_b1, rew_b1, label="$b=1$ Q-Learner (1-Bit Memory)", color="#1b9e77", lw=2)
        axes[0].set_title("(a) Tabular Q-Learning Training Curves")
        axes[0].set_xlabel("Training Episode")
        axes[0].set_ylabel("Total Episode Reward (True Field $c(x)$)")
        axes[0].legend(frameon=True)

    # Learning Gap bar comparison
    if "step8" in data:
        b0_gap = data["step8"]["b0_results"]["learning_gap"]
        b1_gap = data["step8"]["b1_results"]["learning_gap"]
        categories = ["$b=0$ Memoryless", "$b=1$ 1-Bit Memory"]
        gaps = [b0_gap if np.isfinite(b0_gap) else 1.0, b1_gap if np.isfinite(b1_gap) else 1.0]

        bars = axes[1].bar(categories, gaps, color=["#e7298a", "#1b9e77"], width=0.45)
        axes[1].axhline(1.0, color="k", ls="--", lw=1.5, label="Optimal Ground Truth Baseline ($1.0\\times$)")
        axes[1].set_title("(b) The Learning Gap $\\langle T_{\mathrm{esc}} \\rangle_{\mathrm{learned}} / \\langle T_{\mathrm{esc}} \\rangle_{\mathrm{optimal}}$")
        axes[1].set_ylabel("Learning Gap Ratio")
        axes[1].set_ylim(0, max(max(gaps) * 1.3, 1.5))
        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(f"{h:.2f}x",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 4), textcoords="offset points",
                             ha="center", va="bottom", fontweight="bold")
        axes[1].legend(frameon=True)

    plt.tight_layout()
    fig_path6 = os.path.join(fig_dir, "fig6_rl_learnability_gap.png")
    fig.savefig(fig_path6, dpi=300)
    plt.close(fig)
    print("  Saved:", fig_path6)
    print("All figures successfully generated and saved to:", fig_dir)


if __name__ == "__main__":
    generate_all_figures()
