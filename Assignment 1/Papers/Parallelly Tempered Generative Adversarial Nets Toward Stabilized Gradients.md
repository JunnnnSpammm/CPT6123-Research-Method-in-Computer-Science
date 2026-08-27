### Parallelly Tempered Generative Adversarial Nets Toward Stabilized Gradients

---

### Abstract & Overview (Page 1)

* **Core Problem**: Generative Adversarial Networks (GANs) suffer from severe training instability and non-convergence (mode collapse) when approximating multimodal target distributions.


* **New Perspective**: The authors analyze mode collapse through the lens of the critic’s gradient variance.


* **Proposed Solution**: Parallelly Tempered Generative Adversarial Networks (PTGAN) introduce a convex interpolation scheme that creates tempered intermediate distributions to stabilize gradient updates without annealing schedules.


* **Key Contributions**: PTGAN achieves variance reduction, nearly minimax optimal rates, superior sample quality across image and tabular benchmarks, and enables fair synthetic data generation.



---

### Section 1: Introduction (Pages 1–2)

* **GAN Framework**: Consists of a critic $D \in \mathcal{D}$ and generator $G \in \mathcal{G}$ parameterized by neural networks to minimize probability divergence metrics.


* **Prior Stabilization Work**: Previous efforts focused on network architecture tweaks, gradient norm penalties (e.g., Lipschitz constraints), or adding random noise $X + \epsilon$ to expand support.


* **Novelty of PTGAN**: Instead of adding random noise (which only inflates within-mode variance), PTGAN smooths multimodal distributions by convexly interpolating data points, directly reducing between-mode distance and gradient variance.



---

### Section 2: Estimation Mechanism & Gradient Instability (Pages 2–5)

#### 2.1 Neural Distance & Assumptions

* **Objective Metric**: The paper uses the scaled 1-Wasserstein distance under the Kantorovich-Rubinstein duality:

$$d_{\mathcal{D}}(p_X, p_{G(Z)}) = \sup_{D \in \mathcal{D}} \{E[D(X)] - E[D(G(Z))]\}$$


* **Standard Assumptions**:
* **(A1)** Bounded network parameters $\vert{}\vert{}W_i\vert{}\vert{}_F \le M_w(i)$ and $\vert{}\vert{}V_j\vert{}\vert{}_F \le M_v(j)$.
* **(A2)** $K_\kappa$-Lipschitz activation functions (e.g., ReLU, LeakyReLU).
* **(A3)** Bounded support for input $X \in \mathcal{X}$ and latent variable $Z \in \mathcal{Z}$.



#### 2.2 Adversarial Estimation Dynamics

* **Optimization Steps**: Alternates mini-batch gradient updates for weights $w^{(t)}$ (critic) and $v^{(t)}$ (generator) using mini-batch estimator $\hat{L}_b(D, G)$.
* **Support Breakdown**: Target support $\mathcal{S}$ is partitioned into recovered support $\mathcal{S}_t$ and missing support $\mathcal{S}_t^c = \mathcal{S} \setminus \mathcal{S}_t$. Incomplete generator updates force the critic to differentiate $\mathcal{S}_t^c$ from $G^{(t)}(Z)$.

#### 2.3 Theoretical Gradient Analysis under Mode Collapse

* **Proposition 1 (Multimodality Measure)**: The neural distance $d_{\mathcal{D}}(p_1, p_2)$ is bounded by the 1-Wasserstein distance $d_{W_1}(p_1, p_2)$ and captures between-variability $\vert{}\mu_1 - \mu_2\vert{}$ in Gaussian mixtures.
* **Proposition 2 (Lower Bound on Gradient Variance)**: Demonstrates that the covariance norm of the critic’s output layer weights $w_d^{(t)}$ is bounded below by:

$$\left\Vert{} \text{Cov}\left( \frac{\partial \hat{L}_b}{\partial w_d} \right) \right\Vert{}_2 \ge \frac{\zeta_t^\epsilon (1 - \zeta_t^\epsilon) E[L_i^{(t)} \mid L_i^{(t)} > \epsilon]^2}{\Vert{}w_d^{(t)}\Vert{}^2} + \frac{(1 - \zeta_t^\epsilon) \sigma_{\mathcal{S}_t^c}^2 + \sigma_{G^{(t)}}^2}{n_b \Vert{}w_d^{(t)}\Vert{}^2}$$



*Implication*: As the remaining distance $d_{\mathcal{D}}(p_{\mathcal{S}_t^c}, p_{G(Z)})$ increases due to mode collapse, gradient updates become extremely noisy.
* **Theorem 1 (Upper Bound on Gradient Size)**: Establishes that the gradient magnitude w.r.t. internal weights $W_{l,r,c}$ scale directly with $d_{\mathcal{D}}(p_X, p_{G^{(t)}(Z)})$ and auxiliary metric $d_\kappa$. Controlling weight norms prevents exploding gradient variance.

---

### Section 3: Tempered Distributions via Convex Interpolation (Pages 5–6)

* **Convex Interpolation Definition**: For sample pairs $X_{i_1}, X_{i_2} \sim p_X$ and mixing weight $\alpha \sim p_\alpha \in [0, 1]$, an intermediate random variable is defined as:

$$Q_i^\alpha = \alpha X_{i_1} + (1 - \alpha) X_{i_2}$$


* **Proposition 3 (Multimodality Reduction)**: For a 2-component mixture, $Q^\alpha$ contracts the between-mode distance to $\vert{}\mu_1^* - \mu_2^*\vert{} = \frac{3}{4} \vert{}\mu_1 - \mu_2\vert{}$ while adjusting variance, strictly reducing bimodality.
* **Mode Connectivity Advantage (Remark 2)**: Unlike simple additive noise ($X + \epsilon$), convex interpolation creates continuous bridges across disconnected distribution modes in support space $\mathcal{Q}$, significantly diminishing the remaining neural distance.

---

### Section 4: Parallel Estimation Framework (Pages 6–9)

#### 4.1 Joint Optimization & Temperature Distribution

* **Joint Objective**: Generates synthetic data across continuous temperatures by minimizing:

$$\min_{G \in \mathcal{G}} d_{\mathcal{D}}(p_{Q^\alpha, \alpha}, p_{G(Z, \alpha), \alpha})$$


* **Symmetry Transformation**: Enforces physical consistency $G(Z, \alpha) \stackrel{d}{=} G(Z, 1 - \alpha)$ using $t(\alpha) = -2\vert{}\alpha - 0.5\vert{} + 1$.
* **Temperature Mixture Distribution**: Concentrates training on the target data $\alpha=1$ while sampling tempered points via:

$$p_\alpha = r \delta_1(\cdot) + (1 - r) p_{\alpha^*}(\cdot), \quad \alpha^* \sim \text{Unif}(0, 1)$$



#### 4.2 Variance Reduction & Bias-Variance Tradeoff

* **Corollary 1 & Remark 3**: Setting $r < 1$ introduces minor gradient bias for target $p_{X,1}$, but significantly lowers overall gradient variance whenever $d_{\mathcal{D}}(p_{Q^{\alpha^*}, \alpha^*}, p_{G(Z, \alpha^*)}) \le d_{\mathcal{D}}(p_{X,1}, p_{G(Z, 1)})$.

#### 4.3 Coherency Regularization

* **Coherency Penalty ($H$)**: Synchronizes learning rates across different temperature levels to prevent isolated convergence:

$$H = \lambda E_{\alpha_1, \alpha_2, \nu} E_{Q^{\alpha_1}, Q^{\alpha_2}} \left[ \left( \nabla_{Q^{\tilde{\alpha}}} D(Q^{\tilde{\alpha}}, \tilde{\alpha}) \cdot (Q^{\alpha_1} - Q^{\alpha_2}) \right)^2 \right]$$



*Function*: Acts as a weighted distance penalty between modes, constraining weight matrix growth and stabilizing convergence.

#### 4.4 Theoretical Statistical Guarantees

* **Theorem 2 (Estimation Error Upper Bound)**: Shows the empirical neural distance error scales as $O(1/\sqrt{n_e})$, staying size-independent w.r.t. network depth:

$$d_{\mathcal{D}}(p_{Q^\alpha, \alpha}, p_{\hat{G}^*(Z, \alpha), \alpha}) - \inf_{G \in \mathcal{G}} d_{\mathcal{D}} \le C_{UB} \frac{\sqrt{B_X^2 + 1}}{\sqrt{n_e}}$$


* **Proposition 4 (Approximation Error)**: For deep/wide ReLU networks, approximation error decays at rate $O((\lceil V_W/2 \rceil^2 V_D)^{-1/d_X})$.
* **Theorem 3 (Minimax Lower Bound)**: Establishes that PTGAN achieves the optimal global minimax rate $O(1/\sqrt{n})$.

---

### Section 5: Simulation Studies & Empirical Results (Pages 9–12)

#### 5.1 Synthetic Data Generation Performance

* **Image Synthesis (CIFAR10, BloodMnist, CelebA-HQ)**: PTGAN combined with Coherency Penalty (`PT+CP`) outperforms standard regularizations like Spectral Normalization (`SN`), Maximum Penalty (`MP`), and Gradient Penalty (`GP`) across JSD, Pearson $\chi^2$, and Neural Distance losses.

| Dataset | Metric | Standard SN | Max Penalty (MP) | PTGAN (PT+CP) |
| --- | --- | --- | --- | --- |
| **CIFAR10** | Inception Score ($\uparrow$) | 5.591 | 6.929 | **7.292** |
|  | FID ($\downarrow$) | 45.868 | 28.777 | **24.838** |
| **BloodMnist** | Inception Score ($\uparrow$) | 3.711 | 5.006 | **5.071** |
|  | FID ($\downarrow$) | 176.040 | 47.951 | **41.990** |
| **CelebA-HQ** | Multi-Label FID ($\downarrow$) | — | 23.596 | **20.164** |

* **Tabular Data Synthesis (Adult, Law School, Credit Card)**: Measured via downstream model error distance ($S_T$) across Random Forest, SVM, and Logistic Regression. PTGAN consistently converged faster and matched real data distributions closer than `MP`.

#### 5.2 Application: Fair Data Generation

* **FairPTGAN Framework**: By mixing attributes across binary sensitive groups ($A \in \{0,1\}$), setting $\alpha \in (0, 1)$ generates synthetic datasets that break demographic bias while retaining predictive utility.
* **Pareto Frontier Superiority**: Outperforms state-of-the-art fair tabular models (`FairWGANGP`) and preprocessing methods (`GeoRepair`) by achieving lower Statistical Parity (SP) for given utility levels (AUC) without requiring model retraining for each fairness threshold.

---

### Section 6: Discussion (Pages 11–13)

* **Role alongside Diffusion Models**: While diffusion models currently lead in raw visual quality, GANs retain distinct runtime advantages in single-step sampling speed and specialized conditional simulation tasks.
* **Future Directions**: Extending parallel tempering structures to Restricted Boltzmann Machines (RBMs), multi-sample interpolations, and diffusion sampling dynamics.