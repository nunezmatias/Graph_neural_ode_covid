# Graph Neural ODEs for COVID-19 Forecasting: Experimental Summary

**Project Goal:** To develop a physics-informed deep learning model capable of capturing the spatial-temporal dynamics of the COVID-19 epidemic across US states, using mobility networks and Neural Ordinary Differential Equations (ODEs).

This document summarizes the three main experimental phases conducted in this project.

---

## 🏗️ Phase 1: Test 1 - Initial Neural ODE Implementation
**Goal:** Establish the baseline architecture combining Graph Neural Networks (GNN) with ODE solvers.

*   **Architecture:** Basic GNN inside a Neural ODE.
*   **Outcome:** **Unstable.**
    *   The model struggled to converge.
    *   Numerical integration often failed or required extremely small step sizes.
    *   The "Stiff" nature of the differential equations was not properly handled by the initial solver configuration.
*   **Key Learning:** Standard deep learning initialization is too chaotic for differential equations. We needed a curriculum learning approach and better regularization.

---

## 📉 Phase 2: Test 2 - Deterministic Model (Stability & Latent Dynamics)
**Goal:** Stabilize training and achieve accurate curve fitting.

*   **Innovations:**
    *   **Curriculum Learning:** Training on short sequences (5 days) and gradually extending to 20, 40, etc.
    *   **Latent Variables:** Introduced `latent_dim=2` (and later 5) to allow each state to have a comprehensive "health state" vector beyond just Infection/Recovery counts.
    *   **Solver:** Switched to `Tsit5` with careful tolerance tuning.
*   **Outcome:** **Success (with caveats).**
    *   Training Loss converged to ~0.05.
    *   The model perfectly fitted the training data (0-180 days).
    *   **Problem:** It exhibited "Overfitting" and "Overconfidence". It memorized the training noise and provided a single line prediction for the future, failing to express risk.
*   **Key Results:**
    *   See `Resultados/Reporte_Final_Matias.md` for the detailed analysis.
    *   See `Resultados/test-2/` for specific plots (if available).

---

## 🎲 Phase 3: Test 3 - Probabilistic Model (Uncertainty Quantification)
**Goal:** Transform the system into a robust forecasting tool that quantifies risk and uncertainty.

*   **Methodology: Frozen Dropout Ensembles**
    *   We implemented a Bayesian approximation using **Frozen Dropout**.
    *   Unlike standard dropout (which breaks ODE solvers), we sample a mask *once per trajectory*. This creates a "Multiverse" of 100 slightly different valid models.
*   **Outcome:** **Robust Forecasting.**
    *   Training Loss stabilized at ~0.15 (optimizing for robustness over memorization).
    *   **Forecasting:** Successfully predicted the "Second Wave" in unseen data (Days 181-400).
    *   **Uncertainty Bands:** Produced calibrated 95% and 50% confidence intervals.
        *   **Wide Bands:** In peaks (NY, OH), correctly signaling high instability (Butterfly Effect).
        *   **Narrow Bands:** In valleys, correctly signaling stability.
*   **Key Documents & Visuals:**
    *   📄 **Detailed Report (Paper Style):** `Resultados/test-3/Reporte_Incertidumbre_Detallado.md` (Contains mathematical proofs and deep analysis).
    *   🖼️ **Visual Gallery:** `Resultados/test-3/Reporte_Incertidumbre_Dual.html` (Interactive grid of all 50% vs 95% CI plots).
    *   📜 **Methodology:** `Resultados/test-3/README.md`.

---

### 📂 Directory Structure Guide for Reviewers

*   **`Train/`**: Source code for model training (`model_opt.jl`).
*   **`Test/`**: Diagnostic and Analysis scripts.
    *   `uncertainty_analysis_full.jl`: The script that generates the Test 3 dual-band plots.
*   **`Resultados/`**:
    *   `test-3/`: Contains the latest and most advanced findings (Probability/Uncertainty).
    *   `plots/test3_uncertainty_full/`: Raw images of the forecasts.

---

## 🚫 Phase 4: Test 4 - Ablation Study (No Covariates)
**Goal:** Determine the importance of external forcing (mobility, climate) by training the model "blind".

*   **Methodology:**
    *   Removed all covariate inputs ($u_{covars} = \emptyset$).
    *   Relied solely on the autoregressive dynamics of infection counts and the adjacency matrix.
*   **Outcome:** **Failure (Instructive).**
    *   Training loss plateaued at high values (~1000 vs ~0.1 in Test 3).
    *   Even with doubled epochs (3000+) and learning rate annealing, the optimization landscape was too flat/rugged without the "guide" of external factors.
*   **Key Learning:** Covariates are non-negotiable. The epidemiological dynamics are effectively "driven systems"; without the external drivers (mobility), the ODE cannot recover the complex wave patterns.

---

## 🚀 Phase 5: Test 5 - Zero-Shot Generalization
**Goal:** Verify if the Physics-Informed GNN has learned universal dynamics or simply memorized the training states.

*   **Challenge:**
    *   **Train Set:** 10 continuous states (East/Midwest/South).
    *   **Test Set:** 5 completely new states (West/Mountain: WA, AZ, PA, MI, MA) never seen during gradient descent.
    *   **Task:** Predict the pandemic curve for the Test Set without *any* retraining (Zero-Shot).
*   **Methodology:**
    *   **Model Transfer:** Used the robust model from Test 3 (`par_opt_test3.jld2`).
    *   **Latent Initialization:** New states do not have learned latent vectors. We initialized them using the statistical distribution (Mean/Std) of the training latents.
    *   **Spectral Normalization:** The test subgraph had different spectral properties (eigenvalues > 1). We applied spectral normalization to maintain stability.
*   **Outcome:** **Qualitative Success.**
    *   **Timing:** perfectly predicted the *timing* of infection peaks (e.g., Day 250 in WA) purely based on the network propagation from observed neighbors.
    *   **Magnitude:** Scale discrepancies were present (expected due to unoptimized latents), but the shape and phase were correct.
*   **Key Visuals:**
    *   See `Resultados/test-5/report.html` for the interactive summary.


---

## 🧪 Phase 6: Test 6 - Counterfactual Analysis (Causal Reasoning)
**Goal:** Validate that the model has learned **causal relationships**, not just correlations, by simulating policy interventions that never occurred.

*   **Methodology:**
    *   Simulated a lockdown in NY (Day 60-120): 90% reduction in activity inputs + complete graph isolation (edges cut).
    *   Compared three scenarios: Baseline (historical), Lockdown + Gradual Reopening, Permanent Reduction (-20%).
    *   Generated multi-state plots to observe spatial effects on neighboring states.
*   **Outcome:** **Partial Success - Causal Understanding Validated, Spatial Coupling Weak.**
    *   **NY Direct Effects:** Model correctly predicted dramatic case reduction during lockdown and "snap-back" after reopening (forcing-driven dynamics).
    *   **Spatial Effects:** NY's isolation had **minimal impact** on neighbors (FL, IL, NC, NJ) - lines were nearly identical.
*   **Key Learning:** 
    *   ✅ Model understands **causality**: reduced inputs → reduced transmission.
    *   ⚠️ **Network structure contributes minimally** - dynamics are dominated by local inputs and latent features, not spatial diffusion.
    *   Future work: Completed in Phase 7 (Graph Ablation Study).

*   **Key Files:**
    *   `Test/counterfactual_analysis.jl`: Simulation with dynamic graph switching.
    *   `Resultados/test-6/plots/`: NY detailed plot + multi-state comparison.


## 🕸️ Phase 7: Test 7 - Graph Ablation Study (The "Spatial Value" Proof)
**Goal:** Quantify exactly how much the Graph Topology contributes to learning versus just the local ODE dynamics.

*   **Methodology: Three Competitive Universes**
    We trained three identical models side-by-side for 1000 epochs (Deep Fine-Tuning) with `Dropout=0.0`:
    1.  **Full Graph:** Using the real-world mobility network.
    2.  **Isolated:** A graph with no edges (purely local ODEs).
    3.  **Random:** A graph with randomized edges (preserving density but destroying geography).

*   **Outcome:** **Definitive Proof of Spatial Intelligence (15-State Study)**
    *   **Optimization:** The Full Graph and Isolated models optimized with identical efficiency (parallel loss curves), but the Full Graph maintained a **constant superior performance gap** (Loss ~0.064 vs ~0.073).
    *   **Generalization (Zero-Shot):** Tested on 15 unseen states (including complex ones).
        *   **Safety Net Effect:** In difficult states (MD, MN, SC), the Full Graph used neighbors to prevent catastrophe, improving error by **4x to 16x**.
        *   **Contagion Effect:** in some cases (TN), untrained neighbors introduced noise.
        *   **Overall:** The Full Graph is the only architecture capable of preventing "Black Swan" failures.

*   **Key Conclusion:**
    The network structure acts as a **Physical Regularizer**. It prevents the model from hallucinating in complex scenarios by anchoring predictions to the reality of neighbors.

*   **Key Visuals:**
    *   📉 **Convergence:** `Resultados/test-7/plots/convergence/loss_convergence_epoch650.png`
    *   🌍 **Zero-Shot Maps:** `Resultados/test-7/plots/comparative_zeroshot/`

---

## 🌟 Phase 8: Test 8 - The Critical Mass Experiment
**Goal:** Verify if increasing the training graph density from 10 to 25 states stabilizes the GNN and solves the "Spatial Contagion" observed in Test 7.

*   **Methodology:**
    *   **Scale Up:** Increased training set to 25 states (50% of the US graph) to create a "Critical Mass" of trained nodes.
    *   **Evaluation:** Tested on the remaining 25 unseen states (Zero-Shot).
*   **Outcome:** **Definitive Success.**
    *   **Critical Mass Effect:** Trained neighbors effectively "anchor" the predictions for unseen nodes.
    *   **Failures Resolved:** States that failed in Test 7 (PA, TN) became **top performers** in Test 8 due to being surrounded by trained neighbors.
    *   **Topological Limit:** The model only fails for states that are topologically isolated from the training cluster (e.g., WY, SD), proving that performance is a function of graph connectivity.
*   **Key Learnings:**
    *   **Two Mechanisms:** The model generalizes via *Intrinsic Resilience* (weight quality) and *Active Anchoring* (neighbor correction).
    *   **Scale Matters:** 50% graph coverage is sufficient to stabilize the entire continent.
*   **Key Visuals:**
    *   See `Resultados/test-8/README.md` for detailed tables.

---

## 🎯 Phase 9: Test 9 - Graph Characterization & Optimal Holdout
**Goal:** Mathematically characterize the gravity graph and select a topologically representative holdout set to ensure robust validation.

*   **Methodology: Multi-Metric Clustering**
    *   **Features:** Weighted Degree, Betweenness, Eigenvector Centrality, Clustering, and Closeness Centrality.
    *   **Embedding:** PCA (Linear) and UMAP (Nonlinear) parameter sweeps.
    *   **Clustering:** KMeans (k=3) identifying Hubs, Connectors, and Periphery.
*   **Outcome: Strategic Holdout Selection.**
    *   **Representative Set:** Selected 9 states (`AZ, LA, MA, MD, NM, NV, RI, TN, UT`) by sampling 3 from each topological cluster.
    *   **Constraints:** Satisfied geographic diversity (Census Regions) and structural integrity (Fiedler value check).
    *   **Integrity:** Verified that removing holdout nodes does not fracture the training graph connectivity.
*   **Key Visuals:**
    *   🖼️ **Topological MAP:** `Resultados/test-9/pca_clusters.png` (Linear structure).
    *   🌀 **Nonlinear Manifold:** `Resultados/test-9/umap_sweep.png` (Clustering stability).
    *   🗺️ **Geographic Cluster Map:** `Resultados/test-9/cluster_map_us.png` (US states colored by cluster).
    *   📜 **Selection Logic:** `Resultados/test-9/holdout_selection.json`.

