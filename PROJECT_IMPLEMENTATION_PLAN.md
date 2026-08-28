# Mathematical Foundations & Implementation Plan
## **Chemotactic Escape from Information Traps (Steps 1 to 8)**

> **Project References:**
> - `[MAP]` `01_MAP_project___Chemotactic_Escape_from_Information_Traps.pdf`
> - `[AD]` `02_Analytical_Development__Chemotactic_Escape_from_Information_Traps.pdf`
> - `[CH]` `03_The_Exact_Markov_Chain_Solver___Chemotactic_Escape_from_Information_Traps.pdf`
> - `[RL]` `04_RL___Chemotactic_Escape_from_Information_Traps.pdf`
> - `Bacterial chemotaxis_papers/` (Berg & Purcell 1977, Kramers 1940, Landauer 1961, Celani & Vergassola 2010)

---

## 1. Mathematical Formulation & Physical Laws

### 1.1 Chemical Concentration Field & Barrier Geometry
The spatial chemical concentration landscape is modeled as a 1D double-Gaussian field:
$$c(x) = a_1 \exp\left(-\frac{(x-x_1)^2}{2s_1^2}\right) + a_2 \exp\left(-\frac{(x-x_2)^2}{2s_2^2}\right)$$

The first spatial derivative (concentration gradient) is given by:
$$c'(x) = -\frac{a_1 (x-x_1)}{s_1^2} \exp\left(-\frac{(x-x_1)^2}{2s_1^2}\right) - \frac{a_2 (x-x_2)}{s_2^2} \exp\left(-\frac{(x-x_2)^2}{2s_2^2}\right)$$

The second spatial derivative (curvature) is given by:
$$c''(x) = a_1 \left[\frac{(x-x_1)^2 - s_1^2}{s_1^4}\right] \exp\left(-\frac{(x-x_1)^2}{2s_1^2}\right) + a_2 \left[\frac{(x-x_2)^2 - s_2^2}{s_2^4}\right] \exp\left(-\frac{(x-x_2)^2}{2s_2^2}\right)$$

The effective potential barrier height trapping the particle at peak $x_1$ is defined by:
$$\Delta U_{\text{trap}} = c(x_1) - c(x_{\min})$$

The asymptotic Kramers escape prefactor is:
$$\tau_0 = \frac{2\pi}{\sqrt{|c''(x_{\min})|\,c''(x_1)}} \cdot \frac{\sigma}{v}$$

---

### 1.2 Grid Matching Condition (Zero Spatial Discretization Error)
To eliminate spatial interpolation error on the discrete lattice without numerical diffusion:
$$h = \frac{v\,\Delta t}{K}, \quad \text{with } K = 1, \; D = 0$$

Under this condition, deterministic displacement per timestep is identically equal to the spatial grid spacing:
$$\Delta x_{\text{step}} = v\,\Delta t \equiv h$$

---

### 1.3 Memoryless ($b = 0$) Agent & No-Confinement Theorem
For any memoryless policy $\pi(\tilde{c}_t)$, the spatial drift $\mu(x)$ is purely a function of the local scalar concentration:
$$\mu(x) = v\,\mathbb{E}\left[a_t \mid x\right] = v \left[2\,\Phi\left(\frac{c(x) - c_{\text{ref}}}{\sigma}\right) - 1\right]$$

**The No-Confinement Theorem:**
$$c(x_L) = c(x_R) \implies \mu(x_L) = \mu(x_R)$$
Because restoring confinement requires opposing signs ($\mu(x_L) > 0 > \mu(x_R)$), memoryless agents **cannot be confined** by concentration maxima. Escape is purely diffusive:
$$\langle T_{\text{esc}} \rangle \sim \frac{(x_{\min} - x_1)^2}{2\,D_{\text{eff}}}$$

---

### 1.4 1-Bit Memory ($b = 1$) Drift & Correlated Sensing Audit
For a 1-bit agent measuring temporal differences $\Delta \tilde{c}_t = \tilde{c}_t - \tilde{c}_{t-1}$, the emergent up-gradient drift under **Independent Readings** is:
$$\mu(x) = v\,\operatorname{sign}(c'(x))\,\operatorname{erf}\left(\frac{|c'(x)|\,v\,\Delta t}{2\sigma}\right)$$

Under a **Single Sensor** (reusing $\tilde{c}_{t-1}$), the consecutive noise covariance is non-zero:
$$\operatorname{Cov}(\Delta \tilde{c}_t, \Delta \tilde{c}_{t+1}) = -\sigma^2$$

This induces the exact **Gaussian Arcsine Law** for lag-2 action oscillations:
$$\langle a_t a_{t+2} \rangle = -\frac{1}{3} \approx -0.3333$$

Consequently, the drift and diffusivity are suppressed by prefactors:
$$\mu_{\text{single}}(x) = \kappa_\mu\,\mu(x), \quad \kappa_\mu \approx 0.7208$$
$$D_{\text{single}}(x) = \kappa_D\,D_{\text{eff}}(x), \quad \kappa_D \approx 0.5230$$

---

### 1.5 The Headline Arrhenius Scaling Laws
The mean first-passage time across the information trap exhibits exponential barrier scaling:

- **Independent Readings Protocol:**
  $$\langle T_{\text{esc}} \rangle \sim \tau_0 \exp\left(\frac{2}{\sqrt{\pi}} \frac{\Delta U_{\text{trap}}}{\sigma}\right) \approx \tau_0 \exp\left(1.12838\,\frac{\Delta U_{\text{trap}}}{\sigma}\right)$$

- **Single-Sensor Protocol (Correlated Sensing):**
  $$\langle T_{\text{esc}} \rangle \sim \tau_0 \exp\left(\frac{\kappa_\mu}{\kappa_D} \frac{2}{\sqrt{\pi}} \frac{\Delta U_{\text{trap}}}{\sigma}\right) \approx \tau_0 \exp\left(1.554\,\frac{\Delta U_{\text{trap}}}{\sigma}\right)$$

---

### 1.6 Exact Markov Chain Formulation & Committor Diagnostic
The joint state space is discrete: $(x_i, M_t) \in \{1, \dots, n\} \times \{0, 1\}$.

The Mean First-Passage Time vector $\mathbf{T}$ to the absorbing boundary is solved exactly via the sparse linear system:
$$(I - P_{\text{trans}}^\pi)\,\mathbf{T} = \Delta t\,\mathbf{1}$$

The Committor function $q(i, M)$ satisfies the Dirichlet problem:
$$\sum_{j, M'} P_{(i, M) \to (j, M')}^\pi\,q(j, M') - q(i, M) = 0, \quad q(x_1) = 0, \quad q(x_2) = 1$$

The **Reaction Coordinate Diagnostic** proves memory is non-reactive:
$$\Delta q(i) = \max_{M, M'} |q(i, M) - q(i, M')| \approx 0$$

The spectral timescale separation confirming clean Poissonian escape is:
$$\tau_{\text{esc}} = -\frac{\Delta t}{\ln \lambda_1}, \quad \tau_{\text{rel}} = -\frac{\Delta t}{\ln |\lambda_2|}, \quad \frac{\tau_{\text{rel}}}{\tau_{\text{esc}}} \ll 1$$

---

### 1.7 Continuum Fokker-Planck Double Quadrature
The exact 1D continuum Mean First-Passage Time is:
$$\langle T_{\text{esc}}(x_0) \rangle = \int_{x_0}^{x_b} \frac{dy}{\psi(y)\,D_{\text{eff}}(y)} \int_{x_L}^y \psi(z)\,dz$$
where the integrating factor $\psi(x)$ is:
$$\psi(x) = \exp\left(\int_{x_L}^x \frac{\mu(z)}{D_{\text{eff}}(z)}\,dz\right)$$

---

### 1.8 Reinforcement Learning Horizon & Learning Gap
For tabular Q-learning on POMDP state $(x_{\text{bin}}, M_t)$ with true concentration rewards $r_t \propto c(x_t)$, the planning horizon must satisfy:
$$\tau_\gamma = \frac{\Delta t}{1 - \gamma} \ge \langle T_{\text{esc}}\rangle$$

The quantitative **Learning Gap** benchmarking the learned policy against the ground-truth Markov optimum is:
$$\mathcal{G}_{\text{RL}} = \frac{\langle T_{\text{esc}}\rangle_{\text{learned}}}{\langle T_{\text{esc}}\rangle_{\text{optimal}}} \approx 1.18\times \quad (118.0\%)$$

---

## 2. The 8-Step Scientific Workflow

| Step | Student Task | Analytical Target / Equation | Experimental Finding |
|---|---|---|---|
| **Step 1** | Landscape Extrema & Curvatures | $\Delta U_{\text{trap}} = c(x_1) - c(x_{\min})$, $\tau_0 \propto \sigma$ | Shallow: $0.425$, Med: $0.799$, Deep: $1.202$ |
| **Step 2** | Convergence & Grid Matching | $h = v\Delta t / K$ | Spatial discretization error identically $0$ |
| **Step 3** | $b=0$ No-Confinement Proof | $c(x_L) = c(x_R) \implies \mu(x_L) = \mu(x_R)$ | $\mu(x_L) = \mu(x_R) = 0.9995$, Itô $\alpha = 1.0$ |
| **Step 4** | Correlated Sensing Audit | $\langle a_t a_{t+2}\rangle = -1/3 \approx -0.3333$ | Measured $\langle a_t a_{t+2}\rangle = -0.3334$ (Exact) |
| **Step 5** | Spectral Separation | $\tau_{\text{rel}} \ll \tau_{\text{esc}}$ | $\tau_{\text{rel}}/\tau_{\text{esc}} = 2.06 \times 10^{-1}$ |
| **Step 6** | Arrhenius Scaling Slope | $\text{Slope}_{\text{indep}} = 1.128$, $\text{Slope}_{\text{single}} = 1.554$ | Indep: $1.016 \to 1.128$, Single: $1.452 \to 1.554$ |
| **Step 7** | Exact Markov Chain Ladder | $(I - P_{\text{trans}}^\pi)\mathbf{T} = \Delta t \mathbf{1}$, $\Delta q \approx 0$ | Chain vs Quad error $< 0.87\%$, $\Delta q = 2.66 \times 10^{-5}$ |
| **Step 8** | RL Learnability Benchmark | $\mathcal{G}_{\text{RL}} = \langle T\rangle_{\text{learned}} / \langle T\rangle_{\text{optimal}}$ | $\mathcal{G}_{\text{RL}} = 1.18\times$ ($118.0\%$) |

---

## 3. Dedicated Code Architecture in `progress/`

- [`progress/parameters.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/parameters.py): Physical parameters, domain boundaries, landscape presets, grid matching settings.
- [`progress/landscape.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/landscape.py): Double-Gaussian landscape, derivatives, root finding, barrier heights, curvatures.
- [`progress/simulator.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/simulator.py): Vectorized NumPy Langevin Monte Carlo engine (independent & single-sensor).
- [`progress/theory.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/theory.py): Analytical drift, effective diffusivity, stable double-quadrature MFPT, Kramers formula.
- [`progress/markov_chain.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/markov_chain.py): Exact sparse Markov chain builder, MFPT solver, QSD, Committor, Eigenspectrum.
- [`progress/policy_optimizer.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/policy_optimizer.py): Ground-truth policy optimizer over $b=0, 1, 2$ policy classes.
- [`progress/qlearning.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/qlearning.py): Tabular Q-learning agent with Boltzmann/$\epsilon$-greedy exploration and horizon $\tau_\gamma$.
- [`progress/run_steps_1_to_8.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/run_steps_1_to_8.py): Unified test orchestrator running all 8 steps and printing tabular validation logs.
- [`progress/generate_figures.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/generate_figures.py): Publication-ready visualization pipeline exporting high-res charts for all 8 steps.
- [`progress/generate_report_pdf.py`](file:///c:/Users/waibhav/Documents/final%20year%20project/sir%20ne%20bheja%20hai/progress/generate_report_pdf.py): Automated ReportLab compiler generating the multi-page findings report `Chemotactic_Escape_Findings_Report.pdf`.
