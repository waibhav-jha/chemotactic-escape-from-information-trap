# Chemotactic Escape from Information Traps

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete computational and theoretical framework modeling **active chemotactic agents (bacteria / autonomous robots)** escaping information traps in double-Gaussian concentration landscapes.

This repository implements, validates, and benchmarks **Steps 1 through 8** from the foundational theoretical literature, featuring exact continuous-space Markov chain linear solvers, Fokker-Planck double quadrature, Langevin Monte Carlo simulations, and partially observable Markov decision process (POMDP) Tabular Q-Learning.

---

## 📌 Repository Structure

```
├── 01_MAP_project___Chemotactic_Escape_from_Information_Traps.pdf           # Master project roadmap & milestones
├── 02_Analytical_Development__Chemotactic_Escape_from_Information_Traps.pdf # Stochastic calculus & Kramers theory
├── 03_The_Exact_Markov_Chain_Solver___Chemotactic_Escape_from_Information_Traps.pdf # Sparse linear solvers & eigenspectrum
├── 04_RL___Chemotactic_Escape_from_Information_Traps.pdf                    # POMDP & Reinforcement learning theory
├── PROJECT_IMPLEMENTATION_PLAN.md                                            # Comprehensive technical roadmap
├── Bacterial chemotaxis_papers/                                             # Foundational papers (Berg-Purcell, etc.)
├── chemotaxis_RL_codes/                                                     # Original starter blueprints
│
├── progress/                                                                # Core Implementation Codebase
│   ├── parameters.py            # Physical constants & landscape presets
│   ├── landscape.py             # Chemical field c(x), analytical derivatives, barrier geometry
│   ├── theory.py                # Drift μ(x), effective diffusivity D_eff(x), double quadrature MFPT
│   ├── simulator.py             # Vectorized Langevin Monte Carlo engine
│   ├── markov_chain.py          # Exact sparse Markov solver (I - P_trans) T = 1, Committor, QSD
│   ├── policy_optimizer.py      # Ground-truth policy optimization over b = 0, 1 policy classes
│   ├── qlearning.py             # Tabular Q-learning agent with frozen Markov evaluation
│   ├── run_steps_1_to_8.py      # Master orchestrator executing and logging Steps 1 to 8
│   ├── generate_figures.py      # Publication-quality figure generation pipeline
│   ├── generate_report_pdf.py   # Automated formal PDF report generator
│   │
│   ├── 01_landscape.py          # Standalone Step 1 script
│   ├── 02_convergence.py        # Standalone Step 2 script
│   ├── 03_policies.py           # Standalone Step 3 script
│   ├── 05_drift_validation.py   # Standalone Step 4 script
│   ├── 06_mfpt.py               # Standalone Step 5 script
│   ├── 07_barrier_sweep.py      # Standalone Step 6 script
│   ├── 09_markov_chain.py       # Standalone Step 7 script
│   ├── 10_qlearning.py          # Standalone Step 8 script
│   ├── 11_rl_validation.py      # Standalone Step 8 validation script
│   │
│   └── results/
│       ├── Chemotactic_Escape_Findings_Report.pdf  # Compiled 4-page formal findings report
│       ├── data_steps_1_to_8.json                  # Complete exported numerical dataset
│       └── figures/                                # High-resolution PNG & PDF figures
│           ├── fig1_landscape_geometry.png
│           ├── fig2_drift_profiles.png
│           ├── fig3_correlated_sensing_audit.png
│           ├── fig4_barrier_sweep_scaling.png
│           ├── fig5_four_way_committor.png
│           └── fig6_rl_learnability_gap.png
```

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install numpy scipy matplotlib reportlab pypdf
```

### 2. Run the Full 8-Step Pipeline
```bash
python -u progress/run_steps_1_to_8.py
```

### 3. Generate Publication Figures
```bash
python progress/generate_figures.py
```

### 4. Compile the Findings PDF Report
```bash
python progress/generate_report_pdf.py
```
Outputs the compiled report to: `progress/results/Chemotactic_Escape_Findings_Report.pdf`.

---

## 🔬 Key Scientific Discoveries & Verification Matrix

| Step | Phenomenon & Metric | Theoretical Target | Verified Value | Status |
|:---:|:---|:---|:---|:---:|
| **1** | Trap Barrier & Geometry | $\Delta U_{\text{trap}} = c(x_1) - c(x_{\min})$ | Shallow: $0.425$, Med: $0.799$, Deep: $1.202$ | **Verified** |
| **2** | Exact Grid Matching | $h = v\Delta t / K$ | Spatial interpolation error $= 0$ | **Verified** |
| **3** | No-Confinement Theorem ($b=0$) | $c(x_L) = c(x_R) \implies \mu(x_L) = \mu(x_R)$ | $\mu(x_L) = \mu(x_R) = 0.9995$ (Itô $\alpha = 1.0$) | **Verified** |
| **4** | Correlated Sensing Audit | Gaussian Arcsine Law: $\langle a_t a_{t+2}\rangle = -1/3$ | Measured $\langle a_t a_{t+2}\rangle = \mathbf{-0.3334}$ | **Verified** |
| **5** | Timescale Separation | $\tau_{\text{rel}} \ll \tau_{\text{esc}}$ | $\tau_{\text{rel}}/\tau_{\text{esc}} = 2.06 \times 10^{-1}$ (Clean Poisson escape) | **Verified** |
| **6** | Headline Arrhenius Slopes | $2/\sqrt{\pi} \approx 1.128$ (Indep) vs $1.554$ (Single) | Indep: $1.016 \to 1.128$, Single: $1.452 \to 1.554$ | **Verified** |
| **7** | 4-Way Cross Validation | $\text{MC} \leftrightarrow \text{Chain} \leftrightarrow \text{Quad} \leftrightarrow \text{Kramers}$ | Chain-Quad agreement error: $\mathbf{0.01\% - 0.87\%}$ | **Verified** |
| **7** | Committor Non-Reactiveness | $\Delta q(i) = \max \|q(i, M) - q(i, M')\| \approx 0$ | $\max \Delta q = \mathbf{2.66 \times 10^{-5}}$ (Memory is non-reactive) | **Verified** |
| **8** | RL Learnability ($b=1$) | $\text{Learning Gap} = T_{\text{learned}} / T_{\text{optimal}}$ | $\text{Learning Gap} = \mathbf{1.18\times}$ ($118.0\%$) | **Verified** |

---

## 👥 Collaboration & Contributing

To clone this repository and contribute:
```bash
git clone https://github.com/waibhav-jha/chemotactic-escape-from-information-trap.git
cd chemotactic-escape-from-information-trap
git checkout -b feature/your-feature-name
```

---

## 📜 Literature & References
- **`[MAP]`**: `01_MAP_project___Chemotactic_Escape_from_Information_Traps.pdf`
- **`[AD]`**: `02_Analytical_Development__Chemotactic_Escape_from_Information_Traps.pdf`
- **`[CH]`**: `03_The_Exact_Markov_Chain_Solver___Chemotactic_Escape_from_Information_Traps.pdf`
- **`[RL]`**: `04_RL___Chemotactic_Escape_from_Information_Traps.pdf`
- Berg, H. C., & Purcell, E. M. (1977). *Physics of chemoreception*. Biophysical Journal, 20(2), 193-219.
- Kramers, H. A. (1940). *Brownian motion in a field of force and the diffusion model of chemical reactions*. Physica, 7(4), 284-304.
