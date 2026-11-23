# Reinforcement Learning & Stochastic Control Mathematics

This page documents the mathematical formulations behind PyAlloq's RL environments, reward functions, endogenous liquidity impact dynamics, Masked-Dirichlet PPO, continuous-time exploratory mean-variance control, and rough-path regime gating in `pyalloq.rl`.

---

## 1. Multiverse POMDP & Hierarchical Meta-RL Environments

### `POMDPPortfolioEnv`

Models multi-period allocation with an optional cash sleeve \(w_{\text{cash}} = 1 - \mathbf{1}^\top w_{\text{risky}}\) and Point-in-Time tradability mask \(m_t \in \{0, 1\}^N\). Raw agent actions \(a_t\) are projected onto the feasible simplex via one of three modes:

- **Masked Softmax (`"softmax"`)**: \(w_i = \frac{m_i \exp(a_i / \tau)}{\sum_j m_j \exp(a_j / \tau)}\).
- **Masked Dirichlet (`"dirichlet"`)**: \(\alpha_i = (\ln(1 + e^{a_i}) + 1) m_i\), yielding deterministic mean \(w = \alpha / \sum_j \alpha_j\) or stochastic draw \(w \sim \operatorname{Dirichlet}(\alpha)\).
- **ReLU-\(\ell_1\) (`"relu_l1"`)**: \(w_i = \frac{\max(a_i, 0) m_i}{\sum_j \max(a_j, 0) m_j}\).

Portfolio wealth compounds risky holdings plus cash accruing at \(r_f / 252\) net of any `BaseCostModel`.

### `MetaPortfolioEnv`

Given \(K \ge 2\) base allocators emitting candidate weights \(w_t^{(1)}, \dots, w_t^{(K)}\), the meta-agent selects simplex blending weights \(\alpha_t \in \Delta^K\) to form \(w_t^{\text{meta}} = \sum_{k=1}^K \alpha_{t,k} w_t^{(k)}\), stepping forward by `rebalance_stride` days with self-financing weight drift.

---

## 2. Reward Functions (`pyalloq.rl.rewards.functions`)

All reward functions receive realized `net_return` \(r_t^{\text{net}}\), turnover \(\|\Delta w_t\|_1\), unrealized drawdown \(\text{DD}_t\), and Bures-Wasserstein regime distance \(W_2\):

### 1. Differential Sharpe Ratio (`DifferentialSharpeReward`, Moody & Saffell, 2001)

Maintains exponential moving first and second moments \(A_t = A_{t-1} + \eta(r_t^{\text{net}} - A_{t-1})\) and \(B_t = B_{t-1} + \eta((r_t^{\text{net}})^2 - B_{t-1})\):

\[
D_t = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{\max(B_{t-1} - A_{t-1}^2, \, \epsilon)^{3/2}}, \qquad
\mathcal{R}_t = \operatorname{clip}(D_t, -10, 10) - \lambda_{\text{dd}} |\text{DD}_t| - \eta_{W_2} W_2 \|\Delta w_t\|_1
\]

### 2. Stepwise Sortino (`StepwiseSortinoReward`)

Uses rolling downside semi-deviation \(\sigma_- = \sqrt{\frac{1}{W}\sum_{\tau} \min(0, r_\tau^{\text{net}} - \text{MAR})^2 + \sigma_{\text{floor}}^2}\):

\[
\mathcal{R}_t = \operatorname{clip}\!\left(\frac{r_t^{\text{net}} - \text{MAR}}{\sigma_-}, -10, 10\right) - \lambda_{\text{dd}} |\text{DD}_t|
\]

### 3. CARA Log-Wealth Utility (`CARAUtilReward`)

\[
\mathcal{R}_t = \ln(1 + r_t^{\text{net}}) - \frac{\gamma}{2} (r_t^{\text{net}})^2 - \lambda_{\text{dd}} |\text{DD}_t| - \eta_{W_2} W_2 \|\Delta w_t\|_1
\]

---

## 3. Endogenous Liquidity & Graph Contagion Impact (`EndogenousLiquidityImpactWrapper`)

Tracks a transient order-book impact state \(q_t \in \mathbb{R}^N\) with resilience \(\rho \in (0, 1)\) and cross-asset contagion across the normalized symmetric adjacency matrix \(\tilde{W} = D^{-1/2} W D^{-1/2}\):

\[
q_{t+1} = \rho \, q_t + \lambda |\Delta w_t| + \kappa \, \tilde{W} |\Delta w_t|, \qquad
C_t^{\text{endog}} = \langle q_{t+1}, \, |\Delta w_t| \rangle
\]

At every step, \(C_t^{\text{endog}}\) is deducted from both the RL reward and portfolio wealth (`portfolio_value *= max(0, 1 - C_t^endog)`).

---

## 4. Masked-Dirichlet PPO (`SB3PortfolioAgent`)

When `policy_backbone="set_transformer"`, `SB3PortfolioAgent` optimizes `SetTransformerPolicyNet` directly on the active simplex support \(\mathcal{A}_t = \{j : \tilde{m}_{t,j} = 1\}\) using the exact masked Dirichlet log-density (\(\alpha_0 = \sum_{j \in \mathcal{A}_t} \alpha_j\)):

\[
\ln \pi_\theta(w_t \mid s_t) = \ln \Gamma(\alpha_0) - \sum_{j \in \mathcal{A}_t} \ln \Gamma(\alpha_j) + \sum_{j \in \mathcal{A}_t} (\alpha_j - 1) \ln w_{t,j}
\]

with the clipped PPO surrogate objective plus value MSE and differential entropy bonus:

\[
\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}\!\left[\min\!\left(\rho_t(\theta) \hat{A}_t, \, \operatorname{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)\right] + c_v (V_\theta(s_t) - \hat{R}_t)^2 - c_H \mathcal{H}\!\left(\pi_\theta(\cdot \mid s_t)\right)
\]

---

## 5. Continuous-Time Exploratory Mean-Variance (`WonhamRegimeFilter` & `ExploratoryRegimeMVAllocator`)

### Causal Wonham SDE Regime Filter (`WonhamRegimeFilter`)

Given \(K\) Gaussian regimes \(\{(\mu_k, \Sigma_k)\}_{k=1}^K\) ordered by \(\operatorname{Tr}(\Sigma_k)\) and continuous-time Markov generator \(Q = (P - I) / \Delta t\) (\(\sum_j Q_{ij} = 0\)), the regime belief vector \(p_t \in \Delta^K\) evolves causally via the Wonham filter (blended 65/35 with quadratic Gaussian log-likelihood weights for second-order volatility discrimination):

\[
dp_{t,k} = (p_t^\top Q)_k \, dt + p_{t,k} (\mu_k - \bar{\mu}_t)^\top \bar{\Sigma}_t^{-1} (r_t - \bar{\mu}_t)
\]

### Intertemporal Hedging & Entropy-Regularized Exploratory Policy (`ExploratoryRegimeMVAllocator`)

1. **Law of Total Covariance + Jump Risk**:
   \[
   \bar{\Sigma} = \sum_{k=1}^K p_{T,k} \left(\Sigma_k + (\mu_k - \mu_{\text{myopic}})(\mu_k - \mu_{\text{myopic}})^\top\right) + \lambda_{\text{jump}} \Sigma_{\text{jump}}
   \]
2. **Markov Regime-Transition Hedging Demand**: Projects beliefs forward over horizon \(H\) via the matrix exponential \(p_{\text{fwd}} = p_T^\top \exp(Q \cdot H/252)\), giving \(\mu_{\text{total}} = \mu_{\text{myopic}} + \omega_{\text{hedge}}(\mu_{\text{fwd}} - \mu_{\text{myopic}})\).
3. **Optimal Mean Control \(\mu_\pi^*\)**:
   \[
   \mu_\pi^* = \arg\max_{w \in \Delta^N} \; \mu_{\text{total}}^\top w - \frac{\gamma}{2} w^\top \Sigma_{\text{psd}} w - \frac{\kappa_{\text{turn}}}{2} \|w - w_{t^-}\|_2^2
   \]
4. **Analytical Exploratory Covariance \(\Sigma_\pi^*\)**: Under Shannon or Tsallis-\(q\) entropy regularization with temperature \(\lambda_{\text{ent}}\), the optimal Gaussian exploratory policy is \(\pi^*(w \mid s_t) = \mathcal{N}(\mu_\pi^*, \Sigma_\pi^*)\) where:
   \[
   \Sigma_\pi^* = s_q \frac{\lambda_{\text{ent}}}{\gamma} \left(\Sigma_{\text{psd}} + (\kappa_{\text{turn}} + 10^{-4}) I\right)^{-1}, \qquad
   s_q = \begin{cases} 1 & \text{Shannon} \\ \dfrac{2}{1 + q} & \text{Tsallis } (q > 0) \end{cases}
   \]

---

## 6. Rough-Path Signature-MMD Regime Gate (`SignatureMMDRegimeGate`)

To detect non-Gaussian, non-Markovian structural breaks invisible to Gaussian covariance baths, `SignatureMMDRegimeGate` extracts rolling depth-2 Lyndon log-signatures (increments plus antisymmetric Lévy areas \(\mathcal{A}^{ij} = \frac{1}{2}\sum_t (X_{t-1}^i dX_t^j - X_{t-1}^j dX_t^i)\)) over fast vs. slow windows and computes the normalized Signature Maximum Mean Discrepancy:

\[
\operatorname{MMD}_{\text{sig}} = \frac{1}{\sqrt{D}} \left\|\mathbb{E}_{\text{fast}}[\operatorname{LogSig}_2] - \mathbb{E}_{\text{slow}}[\operatorname{LogSig}_2]\right\|_2
\]

Rebalancing weights are damped exponentially whenever \(\operatorname{MMD}_{\text{sig}}\) breaches `mmd_threshold`:

\[
\lambda_{\text{gate}} = \operatorname{clip}\!\left(\exp\!\left(-\beta_{\text{sens}} \max(0, \operatorname{MMD}_{\text{sig}} - \text{mmd\_threshold})\right), \, \lambda_{\min}, \, 1\right), \qquad
w_t^{\text{gated}} = \lambda_{\text{gate}} w_t^{\text{target}} + (1 - \lambda_{\text{gate}}) w_{t^-}
\]
