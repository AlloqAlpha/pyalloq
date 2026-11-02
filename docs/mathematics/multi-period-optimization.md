# Multi-Period Portfolio Optimization (MPO) — Mathematical Foundations

Single-period portfolio optimization (SPO) treats each rebalance date \(t\) as an isolated static problem, ignoring the intertemporal coupling induced by **transaction costs**, **nonlinear market impact**, **multi-horizon alpha decay**, and **future rebalancing flexibility**.

PyAlloq implements a unified **3-Pillar Multi-Period Portfolio Optimization (MPO)** framework bridging:

1. **Classical Convex Model Predictive Control (MPC) & Operator Splitting (ADMM)** (Boyd et al., 2017; Gârleanu & Pedersen, 2013)
2. **Decision-Focused Deep Learning (DFL) & Entropic Mirror-Descent Fixed-Point (MDFP) Layers** (Agrawal et al., 2019; Elmachtoub & Grigas, 2022)
3. **FinRL Deep Reinforcement Learning, Two-Level Bayesian HRP (RL-BHRP), Deep Hedging Trajectory Optimization, & Thermodynamic Regime Gating** (Buehler et al., 2019; Liu et al., 2022; Moroke et al., 2026)

---

## 1. Intertemporal Wealth & Self-Financing Dynamics

Let \(N\) risky assets (plus an optional cash account) have gross simple return vector \(R_{t+1} = \mathbf{1} + r_{t+1} \in \mathbb{R}_{++}^N\) over period \([t, t+1]\).

At the beginning of period \(t\):

- \(w_{t^-} \in \Delta^N\) denotes the **pre-trade drifted weight vector** inherited from period \(t-1\).
- \(z_t \in \mathbb{R}^N\) denotes the **normalized trade vector** executed at time \(t\), resulting in the **post-trade weight vector**:
  \[
  w_t = w_{t^-} + z_t, \qquad \mathbf{1}^\top z_t + \phi_{\text{trade}}(z_t) = 0 \quad (\text{or } \mathbf{1}^\top w_t = 1 \text{ under self-financing rescale}).
  \]
- Over period \([t, t+1]\), market returns cause the post-trade weights \(w_t\) to drift nonlinearly to the next pre-trade state \(w_{(t+1)^-}\):
  \[
  w_{(t+1)^-} = \frac{w_t \circ (\mathbf{1} + r_{t+1})}{1 + r_{p, t+1}^{\text{net}}}, \qquad
  r_{p, t+1}^{\text{net}} = (1 - \phi_{\text{trade}}(z_t))(1 + w_t^\top r_{t+1}) - 1.
  \]

Within a convex lookahead planning horizon \(\tau = t, \dots, t+H-1\) where future returns are small (\(\|r_\tau\|_\infty \ll 1\)), the linearized intertemporal state transition holds:
\[
w_\tau = w_{\tau-1} + z_\tau, \qquad w_{t-1} \equiv w_{t^-}.
\]

---

## 2. Microstructural Transaction Cost & Market Impact Models

At each planning stage \(\tau\), rebalancing trades \(z_\tau = w_\tau - w_{\tau-1}\) incur temporary and permanent execution costs:

\[
\phi_{\text{trade}}(z_\tau) = \underbrace{\left(\kappa_{\text{spread}}^\top |z_\tau|\right)}_{\text{Proportional Bid-Ask } (p=1)}
+ \underbrace{\left(\kappa_{\text{impact}}^\top |z_\tau|^{3/2}\right)}_{\text{Square-Root Market Impact } (p=3/2)}
+ \underbrace{\frac{1}{2} z_\tau^\top \Lambda_\tau z_\tau}_{\text{Quadratic Temporary Cost } (p=2)}.
\]

### 2.1 Proportional Bid-Ask Spread (\(p=1\)) & the No-Trade Zone
The \(\ell_1\) penalty \(\kappa_{\text{spread}}^\top |w_\tau - w_{\tau-1}|\) is non-differentiable at \(z_{\tau,i} = 0\). By first-order subdifferential KKT conditions:
\[
\hat{\mu}_{\tau,i} - \gamma (\Sigma_\tau w_\tau)_i - \nu_\tau \in [-\kappa_{\text{spread},i}, +\kappa_{\text{spread},i}] \implies z_{\tau,i}^* = 0.
\]
Thus, proportional costs create a rigorous **polyhedral no-trade region** around \(w_{t^-}\): the optimizer holds existing positions untouched unless the marginal risk-adjusted alpha exceeds the half-spread threshold \(\kappa_{\text{spread},i}\).

### 2.2 Square-Root Market Impact (\(p=3/2\)) via Power Cones
Following Almgren & Chriss (2001) and Boyd et al. (2017), executing a fraction of Average Daily Volume (ADV) incurs price impact proportional to \(|z_{\tau,i}|^{1/2}\), yielding total dollar cost proportional to \(|z_{\tau,i}|^{3/2}\):
\[
\kappa_{\text{impact},i} |z_{\tau,i}|^{3/2}, \qquad \kappa_{\text{impact},i} \propto \sigma_i \sqrt{\frac{V_{\text{port}}}{\text{ADV}_i}}.
\]
In `CVXPY`, `cp.power(cp.abs(z_tau), 1.5)` is canonicalized into **3D power cones / second-order cones (SOCP)** solvable in polynomial interior-point iterations.

### 2.3 Gârleanu-Pedersen (2013) Permanent Price Impact & Aim Portfolios
When large institutional trades move the equilibrium market price permanently, an intertemporal price-distortion state \(q_\tau \in \mathbb{R}^N\) evolves with resilience rate \(\rho_{\text{perm}} \in (0, 1]\):
\[
q_\tau = (1 - \rho_{\text{perm}}) q_{\tau-1} + \eta_{\text{perm}} z_\tau, \qquad q_{t-1} = \mathbf{0},
\]
imposing an additional stage execution drag \(\frac{1}{2} q_\tau^\top q_\tau\). Under quadratic costs and exponential alpha decay, Gârleanu & Pedersen (2013) prove that the optimal multi-period policy trades partially at constant speed toward an **Aim Portfolio** \(w_t^{\text{aim}}\), which itself is a discounted convex combination of future single-period Markowitz portfolios.

---

## 3. Multi-Horizon Alpha Decay & Term Structure

Alpha signals decay at heterogeneous rates across horizons \(h = 1, \dots, H\). Let \(K\) signal families (e.g., short-term reversal, medium-term momentum, slow value/carry) have half-lives \(H_{1/2}^{(k)}\) and decay rates \(\lambda_k = \frac{\ln 2}{H_{1/2}^{(k)}}\).

The expected return forecast at horizon step \(h \in \{0, \dots, H-1\}\) is:
\[
\hat{\mu}_{t+h|t} = \sum_{k=1}^K \exp(-\lambda_k h) \, s_{t}^{(k)}.
\]
Implemented in [`AlphaDecayReturnEstimator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/classical/estimators/returns/multi_horizon.py), this prevents fast-decaying signals from triggering high-turnover trades that cannot amortize round-trip execution costs.

---

## 4. Pillar 1: Classical Convex MPC & ADMM Operator Splitting

### 4.1 Deterministic Receding-Horizon MPC Formulation
At each rebalance date \(t\), [`MultiPeriodMPCAllocator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/classical/optimizers/multi_period.py) solves the \(H\)-stage convex optimization problem:

\[
\begin{aligned}
\max_{\{w_\tau, z_\tau\}_{\tau=t}^{t+H-1}} \quad & \sum_{\tau=t}^{t+H-1} \beta^{\tau-t} \left[
\hat{\mu}_{\tau|t}^\top w_\tau
- \frac{\gamma}{2} w_\tau^\top \hat{\Sigma}_{\tau|t} w_\tau
- \phi_{\text{trade}}(z_\tau)
- \phi_{\text{hold}}(w_\tau)
\right]
- \frac{\gamma_{\text{term}}}{2} (w_{t+H-1} - w_{\text{bench}})^\top \hat{\Sigma}_{t+H-1|t} (w_{t+H-1} - w_{\text{bench}}) \\
\text{subject to} \quad
& w_\tau = w_{\tau-1} + z_\tau, \qquad w_{t-1} = w_{t^-}, \\
& \mathbf{1}^\top w_\tau = 1, \qquad w_{\min} \le w_\tau \le w_{\max}, \\
& \|z_\tau\|_1 \le \tau_{\max} \quad \forall \tau \in \{t, \dots, t+H-1\}.
\end{aligned}
\]

By the **Receding-Horizon Principle**, only the immediate stage decision \(w_t^*\) is executed in the market; at \(t+1\), the horizon rolls forward with updated pre-trade weights \(w_{(t+1)^-}\).

### 4.2 High-Speed Operator Splitting via ADMM
For large universes or high-frequency backtests where interior-point solvers become a bottleneck, `MultiPeriodMPCAllocator(backend="admm")` reformulates the coupled \(H\)-stage problem using **Alternating Direction Method of Multipliers (ADMM)**:

1. **Primal Weight & Trade Variable Split:** Introduce auxiliary split variables \(u_\tau = z_\tau = w_\tau - w_{\tau-1}\) and \(v_\tau = w_\tau\) with scaled dual variables \(\eta_\tau, \xi_\tau\).
2. **Block-Tridiagonal KKT Linear System (\(w\)-update):** Solving for the stacked trajectory \(W = (w_t, \dots, w_{t+H-1})\) reduces to a positive-definite block-tridiagonal linear system inverted via block Thomas elimination.
3. **Closed-Form Proximal Soft-Thresholding (\(u\)-update):** The non-smooth \(\ell_1\) bid-ask cost and turnover ball decouple into elementwise soft-thresholding:
   \[
   \mathcal{S}_{\kappa / \rho}(x) = \operatorname{sign}(x) \max(|x| - \kappa/\rho, \, 0),
   \]
   followed by Euclidean projection onto the \(\ell_1\)-turnover ball.
4. **Capped Simplex Projection (\(v\)-update):** Projects each \(v_\tau\) onto \(\{w \in \mathbb{R}^N : \mathbf{1}^\top w = 1, \, w_{\min} \le w \le w_{\max}\}\) in \(O(N \log N)\) time.

### 4.3 Scenario-Tree Stochastic MPC with CVaR Tail Control
When `ScenarioMarketData` provides \(M\) Monte Carlo paths \(R^{(m)} \in \mathbb{R}^{H \times N}\) with probabilities \(p_m\) (e.g., from `GARCHEVTCopulaGenerator` or `EntropyPooler`), [`StochasticScenarioMPCAllocator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/classical/optimizers/stochastic_mpc.py) enforces **non-anticipativity** at \(t=0\) and optimizes expected multi-stage return minus Rockafellar-Uryasev **Conditional Value-at-Risk (\(\text{CVaR}_\alpha\))** of cumulative path loss \(L_m\):
\[
\text{CVaR}_\alpha(L) = \min_{\zeta \in \mathbb{R}, \, u_m \ge 0} \left\{ \zeta + \frac{1}{1 - \alpha} \sum_{m=1}^M p_m u_m \;\middle|\; u_m \ge L_m - \zeta \right\}.
\]

---

## 5. Pillar 2: Decision-Focused Learning (DFL) & Differentiable MDFP Layers

### 5.1 The Objective Mismatch of Two-Stage "Predict-Then-Optimize"
Training a neural network exclusively on Mean Squared Error \(\mathcal{L}_{\text{MSE}}(\hat{\mu}, y) = \|\hat{\mu} - y\|_2^2\) treats all return forecast errors equally. However, inside a constrained multi-period optimizer:

- Errors on assets deep inside the **no-trade zone** have zero impact on portfolio weights.
- Small sign errors on assets near the simplex boundary or with low covariance can trigger massive, costly turnover cascades across stages \(t, \dots, t+H-1\).

### 5.2 Entropic Mirror-Descent Fixed-Point (MDFP) Optimization Layer
To embed multi-period optimization directly inside PyTorch computation graphs without expensive CPU-bound conic solvers, [`MDFPOptimizationLayer`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/layers/mdfp.py) solves the coupled \(H\)-stage simplex problem using **Entropic Mirror Descent** (Exponentiated Gradient) under the Kullback-Leibler Bregman divergence \(D_{\text{KL}}(z \| z')\):

\[
z_{s,i}^{(m+1)} = \frac{z_{s,i}^{(m)} \exp\!\left(-\eta \, g_{s,i}^{(m)}\right)}{\sum_{j=1}^N z_{s,j}^{(m)} \exp\!\left(-\eta \, g_{s,j}^{(m)}\right)} \equiv \mathcal{F}_{s,i}\!\left(z^{(m)}, \theta\right),
\]
where \(g_s^{(m)} = \nabla_{z_s} J(z^{(m)}; \theta)\) is the stage gradient incorporating expected returns, covariance risk \(\gamma \Sigma_s z_s\), and smoothed inter-stage turnover coupling \(\kappa \frac{z_s - z_{s-1}}{\sqrt{(z_s - z_{s-1})^2 + \epsilon^2}}\).

### 5.3 \(O(1)\)-Memory Implicit Fixed-Point Differentiation
Rather than unrolling \(M\) mirror-descent steps in autograd memory, `MDFPOptimizationLayer` differentiates through the converged fixed point \(z^* = \mathcal{F}(z^*, \theta)\) via the **Implicit Function Theorem**:
\[
z^*(\theta) = \mathcal{F}(z^*(\theta), \theta) \implies
\left(I - \frac{\partial \mathcal{F}}{\partial z^*}\right) \frac{\partial z^*}{\partial \theta} = \frac{\partial \mathcal{F}}{\partial \theta}.
\]
In the backward pass, given upstream gradient \(v = \frac{\partial \mathcal{L}}{\partial z^*}\), we solve the adjoint linear system via a damped **Neumann Series** of Vector-Jacobian Products (VJPs) evaluated exclusively at \(z^*\):
\[
u^\top = v^\top \left(I - \lambda \frac{\partial \mathcal{F}}{\partial z^*}\right)^{-1} = \sum_{k=0}^K v^\top \left(\lambda \frac{\partial \mathcal{F}}{\partial z^*}\right)^k, \qquad
\frac{\partial \mathcal{L}}{\partial \theta} = u^\top \frac{\partial \mathcal{F}(z^*, \theta)}{\partial \theta}.
\]

### 5.4 SPO+ Surrogate & Hybrid DFL Losses
PyAlloq provides two decision-focused training objectives in [`dfl.py`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/loss_functions/dfl.py):

1. **Multi-Period SPO+ Convex Surrogate Loss** ([`SPOPlusLoss`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/loss_functions/dfl.py)):
   \[
   \mathcal{L}_{\text{SPO}+}(\hat{\mu}, y) = J\!\left(z^*(2\hat{\mu} - y); \, 2\hat{\mu} - y\right) + 2 \langle \hat{\mu}, \, z^*(y) - z^*(2\hat{\mu} - y) \rangle - J(z^*(y); y).
   \]
2. **Hybrid Decision-Focused Loss** ([`HybridDFLLoss`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/dl/loss_functions/dfl.py)):
   \[
   \mathcal{L}_{\text{Hybrid}} = \alpha \, \mathcal{L}_{\text{MPO}}\!\left(z^*(\hat{\mu}), y\right) + (1 - \alpha) \, \|\hat{\mu} - y\|_2^2, \qquad \alpha \in [0, 1],
   \]
   combining multi-period net-of-cost decision utility with MSE regularization to stabilize representation learning in low-SNR financial regimes.

---

## 6. Pillar 3: FinRL DRL, RL-BHRP, Deep Hedging & Thermodynamic Gates

### 6.1 FinRL Markov Decision Process (MDP) & Turbulence Circuit Breaker
In [`PortfolioGymEnv`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/rl/envs/portfolio_env.py), multi-period portfolio management is cast as a continuous-control MDP \((\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)\):

- **State \(s_t\):** Concatenation of pre-trade drifted weights \(w_{t^-}\), lookback return tensor \(R_{t-L+1:t}\), rolling volatilities \(\sigma_t\), Mahalanobis **Turbulence Index** \(d_t\), and Bures-Wasserstein regime distance \(W_2\):
  \[
  d_t = \frac{1}{N} (r_t - \hat{\mu}_t)^\top \hat{\Sigma}_t^{-1} (r_t - \hat{\mu}_t).
  \]
- **Reward \(r_t\):** Differential log net wealth growth penalized by quadratic return variance and thermodynamic turnover friction:
  \[
  r_t = \ln(1 + r_{p,t}^{\text{net}}) - \frac{\gamma}{2} (r_{p,t}^{\text{net}})^2 - \eta_{W_2} W_2(t) \|w_t - w_{t^-}\|_1.
  \]

### 6.2 Two-Level Bayesian Hierarchical Risk Parity (RL-BHRP)
[`BayesianHRPAllocator`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/ml/optimizers/bhrp.py) resolves the return-insensitivity of classical HRP without inverting ill-conditioned \(N \times N\) covariance matrices:

1. **Intra-Cluster Level:** Hierarchical clustering partitions the \(N\) assets into \(K\) correlation clusters, and recursive bisection computes risk-parity intra-cluster weights \(W_{\text{intra}} \in \mathbb{R}^{N \times K}\).
2. **Inter-Cluster Bayesian Level:** Cluster-level equilibrium implied returns \(\pi_C = \Sigma_C b_{\text{hrp}}\) are blended with sample/predicted cluster views \(\mu_C^{\text{view}}\):
   \[
   \bar{\mu}_C = (1 - \tau) \pi_C + \tau \mu_C^{\text{view}},
   \]
   and cluster budgets are tilted toward high-posterior-Sharpe clusters while damping cluster turnover against \(w_{t^-}\).

### 6.3 Deep Hedging Pathwise Trajectory Optimization
Rather than relying on noisy TD(0) Bellman bootstrapping, [`DeepHedgingTrajectoryTrainer`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/rl/trajectory.py) unrolls [`TrajectoryPolicyNet`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/rl/trajectory.py) \(\pi_\theta(R_{t-L+1:t}, w_{t^-})\) along \(H\)-step sample paths \(\omega_j\) and minimizes the **Entropic Risk Measure** of cumulative net wealth via direct pathwise backpropagation:
\[
\mathcal{L}_{\text{DH}}(\theta) = \frac{1}{\gamma} \ln \left( \frac{1}{M} \sum_{j=1}^M \exp\!\left( -\gamma \sum_{s=0}^{H-1} r_{p, t+s}^{\text{net}}(\omega_j; \theta) \right) \right).
\]

### 6.4 Thermodynamic & Bures-Wasserstein Regime Gate (Moroke et al., 2026)
[`WassersteinRegimeGate`](file:///Users/siddesh/Downloads/experiments/public-repositories/pyalloq/src/pyalloq/rl/thermodynamics.py) models the portfolio rebalancing cycle as a non-equilibrium stochastic heat engine coupled to a fast market bath \(\mathcal{N}(\mu_{\text{fast}}, \Sigma_{\text{fast}})\) at temperature \(T_{\text{fast}} = \frac{1}{N}\operatorname{Tr}(\Sigma_{\text{fast}})\) and a slow structural bath \(\mathcal{N}(\mu_{\text{slow}}, \Sigma_{\text{slow}})\) at temperature \(T_{\text{slow}} = \frac{1}{N}\operatorname{Tr}(\Sigma_{\text{slow}})\):

1. **Bures-Wasserstein 2-Distance:**
   \[
   W_2^2(\mathcal{N}_{\text{fast}}, \mathcal{N}_{\text{slow}}) = \|\mu_{\text{fast}} - \mu_{\text{slow}}\|_2^2 + \operatorname{Tr}\!\left(\Sigma_{\text{fast}} + \Sigma_{\text{slow}} - 2\left(\Sigma_{\text{fast}}^{1/2} \Sigma_{\text{slow}} \Sigma_{\text{fast}}^{1/2}\right)^{1/2}\right).
   \]
2. **Carnot Efficiency Bound:**
   \[
   \eta_{\text{Carnot}} = 1 - \frac{\min(T_{\text{fast}}, T_{\text{slow}})}{\max(T_{\text{fast}}, T_{\text{slow}})}.
   \]
3. **Thermodynamic Rebalance Gating:** When \(W_2 > W_{2,\text{thresh}}\) indicates a structural regime rupture where turnover dissipation exceeds extractable alpha, trades are exponentially damped toward existing holdings \(w_{t^-}\):
   \[
   w_t^{\text{gated}} = \lambda_{\text{gate}}(W_2) \, w_t^{\text{target}} + \big(1 - \lambda_{\text{gate}}(W_2)\big) \, w_{t^-}, \qquad
   \lambda_{\text{gate}}(W_2) = \exp\!\left(-\beta_{\text{sens}} \max(0, W_2 - W_{2,\text{thresh}})\right).
   \]
