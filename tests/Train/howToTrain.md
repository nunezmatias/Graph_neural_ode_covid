# Methodological Guide: Multi-Phase Curriculum Training for Geographic GNN-ODEs

## Abstract

Training a continuous-time Graph Neural Ordinary Differential Equation (GNN-ODE) to simulate 400 days of spatial-temporal dynamics across 40 highly coupled geographic nodes is inherently unstable. The continuous dynamics are governed by:

$$ \frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), \mathbf{X}_{cov}(t), \mathcal{G}, t) $$

where $\mathbf{h}(t)$ represents the latent pandemic state, $\mathbf{X}_{cov}$ the covariates, and $\mathcal{G}$ the geographic topology [1]. Direct end-to-end integration over long temporal horizons typically results in vanishing or exploding gradients [2] due to the chaotic nature of biological spreading. 

To achieve convergence, we employ a **Curriculum Learning Strategy** [3]. This progressive 6-phase methodology systematically transitions the network's objective function from short-horizon temporal stabilization to global topological manifold convergence. 

---

## Phase 1: Temporal Curriculum Chunking (Manifold Anchoring)

### Scientific Intuition
Initializing the network with randomized topological weights and demanding a full $t \in [0, 400]$ trajectory immediately forces the integration error to compound exponentially. According to chaotic dynamical systems theory, the divergence of nearby trajectories is bounded by the largest Lyapunov exponent $\lambda$:
$$ \delta \mathbf{h}(t) \approx \delta \mathbf{h}(0) e^{\lambda t} $$

To bound this numerical divergence, we constrain the temporal integration horizon, creating a series of overlapping initial value problems. The loss function is computed locally over temporal windows of size $W$:

$$ \mathcal{L}_{chunk} = \frac{1}{N \times W} \sum_{i=1}^{N} \sum_{\tau=t_0}^{t_0+W} \left| u_i(\tau) - \hat{u}_i(\tau) \right| $$

The optimization architecture expands the temporal window size ($W$) dynamically:
1. **Epochs 1-20 (The Anchor):** $W = 15$ days. The optimizer resolves ultra-local, immediate dynamics.
2. **Epochs 21-35 (The Bridge):** $W = 30$ days. The integration bounds are stretched, forcing the GNN to chain adjacent local phenomena.
3. **Epochs 36-50 (The Assembly):** $W = 45$ days. The network stabilizes multi-wave trajectories within the local temporal chunks.

**Spatial Regularization (Frozen Dropout):** A static spatial dropout mask ($p=0.05$) is applied during integration ($\tilde{\mathcal{G}} = \mathcal{G} \odot \mathbf{M}$) [4]. This forces the GNN to learn a distributed representation of the adjacency matrix $\mathbf{A}$, preventing catastrophic overfitting to high-density nodes (e.g., NY, CA).

### Hyperparameters & Configuration
* **Optimizer:** AdamW (Weight Decay penalizes aggressive initialization)
* **Learning Rate ($\eta$):** $5 \times 10^{-4}$
* **Weight Decay ($\lambda$):** $1 \times 10^{-4}$
* **Epochs:** 50
* **Dropout:** 0.05 (Static/Frozen)

### Execution & Reproducibility
* **Standalone Execution:** `julia --project=. tests/Train/train_40_fast.jl`
* **Output Checkpoint:** `tests/Train/checkpoints/params_40s_fast.jld2`

---

## Phase 2: Global Splicing (Continuous Constraints)

### Scientific Intuition
Phase 2 eliminates the localized boundary conditions ($t_0$). The network is now forced to integrate the contiguous domain $t \in [0, 400]$ strictly from the universal initial condition $\mathbf{u}(0)$. Because the hidden representations were topologically stabilized in Phase 1, the global integration does not explode. Instead, the optimizer surgically splices the previously learned local trajectory chunks into a structurally coherent vector field. The Frozen Dropout ($p=0.05$) remains rigorously enforced.

### Hyperparameters & Configuration
* **Optimizer:** Adam
* **Learning Rate ($\eta$):** $1 \times 10^{-5}$
* **Epochs:** 40
* **Dropout:** 0.05 (Static/Frozen)

### Execution & Reproducibility
* **Standalone Execution:** `julia --project=. tests/Train/resume_phase2.jl`
* **Input Checkpoint:** `params_40s_fast.jld2`
* **Output Checkpoint:** `params_40s_phase2.jld2`

---

## Phase 3: Cosine Annealing (Basin Hunting)

### Scientific Intuition
Upon successful 400-day stabilization, standard gradient descent methodology often stagnates within suboptimal local minima due to the highly fractured spatial-temporal loss landscape. To execute evasive topological traversal, the optimizer is switched to **Stochastic Gradient Descent with Warm Restarts (SGDR)** [5].

The learning rate is modulated by a Cosine Annealing schedule:
$$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{period}}\pi\right)\right) $$

The learning rate decays asymptotically over $T_{period}$ epochs, gently lowering the parameters into the nearest local minimum. The learning rate then instantaneously spikes (warm restart), generating kinetic momentum that ejects the weights from brittle valleys into deeper, more globally optimal optimization basins.

### Hyperparameters & Configuration
* **Optimizer:** Adam (with SGDR Schedule)
* **Learning Rate Range:** $\eta_{max} = 5 \times 10^{-5} \to \eta_{min} = 1 \times 10^{-6}$
* **Cycle Period ($T_{period}$):** 15 Epochs
* **Total Cycles:** 2 (30 Epochs total)
* **Dropout:** 0.05 (Static/Frozen)

### Execution & Reproducibility
* **Standalone Execution:** `julia --project=. tests/Train/resume_phase3.jl`
* **Input Checkpoint:** `params_40s_phase2.jld2`
* **Output Checkpoint:** `params_40s_phase3.jld2`

---

## Phase 4: Deterministic Extrapolation (Hyper-Refinement)

### Scientific Intuition
Phase 3 establishes convergence within a resilient global minimum. Phase 4 initiates the fine descent toward the absolute mathematical floor of this established basin. To accomplish this, **Spatial Dropout is strictly eliminated ($p=0.0$)**.

Removing the stochastic dropping collapses the probabilistic parameter space into a perfectly stationary, deterministic manifold. In this noise-free environment, the vector field is sculpted purely by continuous adjoint gradients [1]. The SGDR parameters are transitioned to a micro-domain to prevent overshooting the basin floor.

### Hyperparameters & Configuration
* **Optimizer:** Adam (with SGDR Schedule)
* **Learning Rate Range:** $\eta_{max} = 2 \times 10^{-5} \to \eta_{min} = 5 \times 10^{-7}$
* **Cycle Period ($T_{period}$):** 20 Epochs
* **Total Cycles:** 2 (40 Epochs total)
* **Dropout:** 0.00 (Disabled)

### Execution & Reproducibility
* **Standalone Execution:** `julia --project=. tests/Train/resume_phase4.jl`
* **Input Checkpoint:** `params_40s_phase3.jld2`
* **Output Checkpoint:** `params_40s_phase4_ext.jld2`

---

## Phases 5 & 6: Ultimate Convergence (The Asymptotic Floor)

### Scientific Intuition
The final protocol extracts the absolute maximum spatial-temporal causality permissible under the static graph $\mathcal{G}$. All cyclical learning rate schedules are deactivated. The model is subjected to relentless gradient sweeps utilizing infinitesimal static learning rates. 

This flatlining of the loss curve formally proves topological saturation: the GNN-ODE has synthesized all available mechanical information within the governing spatial boundaries.

### Hyperparameters & Configuration
* **Optimizer:** Adam (Static)
* **Learning Rate ($\eta$):** $5 \times 10^{-6}$ (Phase 5) $\to 1 \times 10^{-6}$ (Phase 6)
* **Epochs:** 100+ (Terminates dynamically via early stopping)
* **Dropout:** 0.00 (Disabled)

### Execution & Reproducibility
* **Standalone Execution:** `julia --project=. tests/Train/resume_phase5.jl` and `resume_phase6.jl`
* **Input Checkpoints:** `params_40s_phase4_ext.jld2` $\to$ `params_40s_phase5_final.jld2`
* **Output Checkpoints:** `params_40s_phase5_final.jld2` and `params_40s_phase6_final.jld2`

---

## 🚀 The Automated Pipeline Systems

Manually executing scripts in sequence and porting JLD2 checkpoints is error-prone. The repository provides two algorithmic systems to automate the entire 6-phase curriculum.

### 1. The Unified In-Memory Trainer (`auto_train_upgradedTotest.jl`) - RECOMMENDED
This system represents a structural upgrade. Instead of loading the entire global dataset 6 disparate times, this script bootstraps the data into the environment once and executes all 6 curriculum phases sequentially within a contiguous block of RAM. 

*   **Continuous Checkpointing:** Parameter weights (`current_ps`) are passed natively between optimization routines, circumventing disk I/O bottlenecks. Phase transitions are saved automatically to `tests/Train/checkpoints_upgraded/params_upgraded_phaseX.jld2` to prevent overwriting legacy experiments.
*   **Dynamic Early Stopping:** The script evaluates the $L_1$ Mean Absolute Error over a rolling patience window. If the difference falls below $\Delta = 1 \times 10^{-4}$, the phase is halted prematurely to preserve computational resources.

**How to Execute:**
```bash
# Execute deep mathematical optimization (Full Curriculum)
julia --project=. tests/Train/auto_train_upgradedTotest.jl

# Execute analytical compilation test (2 epochs/phase, validates gradient graph)
julia --project=. tests/Train/auto_train_upgradedTotest.jl --smoke-test
```

### 2. The Legacy Sequential Wrapper (`auto_train_pipeline.jl`)
This wrapper script programmatically intercepts the individual file contents of `train_40_fast.jl`, `resume_phase2.jl`, etc., dynamically modifies their local epoch loop variables via regular expressions, writes them to temporary files, and executes them as separate Julia processes.

*   This approach guarantees the exact environment tear-down and reboot behavior present in the historic iterations of Test-8BIS.
*   It supports parametric injection, allowing you to explicitly dictate the epoch counts.

**How to Execute:**
```bash
# Execute the original historic sequence targeting maximum epochs
julia --project=. tests/Train/auto_train_pipeline.jl --full-run

# Isolate execution by parameterizing specific phase boundaries
julia --project=. tests/Train/auto_train_pipeline.jl --phase1-ep=30 --phase2-ep=20 --phase3-ep=30 --phase4-ep=40 --phase5-ep=50 --phase6-ep=50
```

---

## References

1. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. *Advances in neural information processing systems*, 31.
2. Hochreiter, S. (1998). The vanishing gradient problem during learning recurrent neural nets and problem solutions. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, 6(02), 107-116.
3. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th annual international conference on machine learning* (pp. 41-48).
4. Rong, Y., Huang, W., Xu, T., & Huang, J. (2019). Dropedge: Towards deep graph convolutional networks on node classification. *International Conference on Learning Representations (ICLR)*.
5. Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*.
