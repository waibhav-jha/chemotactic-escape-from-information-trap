# 2-Bit Memory ($b = 2$) Chemotaxis Reinforcement Learning Report

This folder serves as the official technical report on extending the Chemotactic Escape POMDP Reinforcement Learning framework to **2-bit temporal memory ($b = 2$)**, highlighting the direct comparison with **1-bit memory ($b = 1$)**.

## Report Package Directory
- **Primary PDF Report:** [`RL_2Bit_Memory_Report.pdf`](RL_2Bit_Memory_Report.pdf) *(3-page publication-quality technical report)*
- **Head-to-Head Comparison Figure:** [`fig_1bit_vs_2bit_comparison.png`](fig_1bit_vs_2bit_comparison.png) *(Escape time bar chart & Committor splitting curve)*
- **Training & Efficiency Figure:** [`fig_rl_2bit_comparison.png`](fig_rl_2bit_comparison.png) *(RL learning curves & learnability gap)*
- **1-Bit vs 2-Bit Comparison CSV:** [`comparison_1bit_vs_2bit.csv`](comparison_1bit_vs_2bit.csv) *(Direct head-to-head metrics)*
- **All 16 Policies CSV:** [`policy_benchmarks_2bit.csv`](policy_benchmarks_2bit.csv) *(Exhaustive evaluation of all deterministic 2-bit policies)*
- **Generator Script:** `progress/generate_2bit_report_pdf.py`

## 1. Problem Statement: Why 1-Bit Fails & Why 2-Bit is Required
Active particles navigate an asymmetric double-Gaussian concentration landscape without spatial GPS coordinates. Under realistic measurement noise ($\sigma = 0.20$), single temporal differences $\Delta c_t = c_t - c_{t-1}$ yield frequent false negative signs even during upward motion. The standard 1-bit heuristic mandates an immediate tumble on negative readings, aborting runs and trapping the agent for $\langle T_{\text{esc}} \rangle = 6,708.85\,\text{s}$.

A 2-bit memory ($b = 2$) tracks the last two temporal increments, providing a temporal filter that distinguishes genuine descents from transient single-step noise dips.

## 2. Direct Head-to-Head Comparison: 1-Bit vs. 2-Bit Memory

| Evaluation Dimension | 1-Bit Memory ($b = 1$) | 2-Bit Learned (PRPR) | 2-Bit Optimum (RPPP) | Physical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Internal States** | 2 (Fall, Rise) | 4 (FF, FR, RF, RR) | 4 (FF, FR, RF, RR) | Doubles temporal resolution without GPS leaks |
| **Decision Policy** | `RRPP` (Tumble on drop) | `PRPR` (Filter single drop) | `RPPP` (Tumble on double drop) | Filters noise dips from true descents |
| **Noise Sensitivity** | Trapped (~48% false aborts) | Noise-Filtered (~81% save) | Optimal Filter | Eliminates premature run abortion |
| **Exact Escape Time** | **6,708.85 s** (~1.86 hrs) | **1,275.82 s** (5.26x speedup) | **288.27 s** (23.27x speedup) | Slashes escape time by 81.0% to 95.7% |
| **Committor Gap $\max \Delta q$** | $9.10 \times 10^{-6}$ (Dormant) | **$1.024 \times 10^{-2}$** (Active) | **$7.146 \times 10^{-2}$** (Dominant) | Memory directly controls escape probability |
| **Reaction Coordinate** | Non-reactive | Active Transition Coord | Dominant Transition Coord | Memory enters transition state physics |
| **Physical Regime** | Arrhenius Barrier Trap | Noise-Filtered Chemotaxis | Optimal Directed Chemotaxis | Qualitative phase shift in transport dynamics |

## 3. Key Physical Discoveries
1. **$5.26\times$ Escape Acceleration:** Tabular Q-learning autonomously discovers the `PRPR` policy, slashing escape time from 6,708.85 s to 1,275.82 s. The theoretical global optimum (`RPPP`) achieves 288.27 s ($23.27\times$ faster).
2. **Emergence of Memory in the Reaction Coordinate:** The Committor difference $\max \Delta q$ surges from $9.10 \times 10^{-6}$ at $b=1$ to $1.024 \times 10^{-2}$ at $b=2$ (and $7.15 \times 10^{-2}$ for `RPPP`), demonstrating that memory physically enters the transition state coordinate.
