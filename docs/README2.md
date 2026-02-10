# Technical Test & Analysis Guide

This document provides a detailed technical overview of the validation scripts and analysis pipelines located in the `Test/` directory. It serves as a reference for understanding the implementation challenges (specifically regarding Geometric Deep Learning layers) and the statistical methodologies used for uncertainty quantification.

## 1. Dropout Implementation & Gradient Flow

A critical challenge encountered was the incompatibility between `GNNLux.GNNChain` and standard `Lux.Dropout` layers within a `Zygote` differentiation context. The `GNNChain` expects layers to handle `(g, x)` inputs, but standard Dropout layers operate on array data only, leading to broadcasting errors during the backward pass.

The following scripts demonstrate the investigation and resolution of this issue:

### `dropout_check.jl` (The Failure Case)
-   **Purpose**: Demonstrates the failure mode when using standard `Lux.Dropout` inside a `GNNChain`.
-   **Observation**: The forward pass may succeed (depending on implementation specifics), but the backward pass (gradient calculation via Zygote) consistently fails.
-   **Root Cause**: `GNNChain` attempts to pipe the graph structure `g` through every layer. `Lux.Dropout` does not accept `g`, causing a method signature mismatch or invalid broadcasting attempt.

### `dropout_fix_check.jl` (The Wrapper Solution)
-   **Purpose**: Tests a partial solution using a custom layer wrapper.
-   **Implementation**: Defines a `GraphDropout` layer that accepts `(g, x, ps, st)`, discards `g`, applies the inner dropout to `x`, and returns `(g, x_dropped)`.
-   **Status**: A viable workaround, but potentially brittle if the GNN library changes its internal API logic for chain composition.

### `dropout_explicit_check.jl` (The Recommended Solution)
-   **Purpose**: Validates the robust "Explicit Model" approach used in production.
-   **Implementation**: Instead of relying on `GNNChain`, we define a custom `ExplicitGNN` struct that inherits from `Lux.AbstractLuxLayer`.
-   **Mechanism**: The `(m::ExplicitGNN)(g, x, ps, st)` function manually orchestrates the data flow. It explicitly passes `(g, x)` to Graph Convolution layers and only `(x)` to Dropout layers.
-   **Outcome**: This restores full control over the computational graph, ensuring correct gradient propagation and compatible state management for Monte Carlo sampling.

## 2. Uncertainty Analysis & Forecasting

We use a **Monte Carlo Dropout** approach to quantify epistemic uncertainty. This involves keeping dropout active during inference (training mode) to sample from the approximate posterior predictive distribution.

### `uncertainty_analysis.jl` (Basic MC)
-   **Scope**: A minimal working example for verifying the MC pipeline.
-   **Methodology**:
    1.  Loads the pre-trained model parameters.
    2.  Samples dropout masks *once per trajectory* (Frozen Dropout) to ensure temporal consistency in the Neural ODE integration.
    3.  Runs $N=100$ forward simulations with different random seeds.
    4.  Visualizes simple "spaghetti plots" to show the spread of possible futures.

### `uncertainty_analysis_full.jl` (Production Analysis)
-   **Scope**: The complete analysis pipeline used for final reporting.
-   **Features**:
    -   **Full Timeline**: Simulates from Day 0 to Day 400 (covering both Training and Testing phases).
    -   **Covariate Splines**: Uses cubic splines to interpolate full-range mobility/climate data, ensuring the model sees realistic future covariates during forecasting.
    -   **Statistical Bounds**: Calculates and plots:
        -   **Mean Prediction**: The expected trajectory.
        -   **50% CI (Interquartile Range)**: Darker, high-probability band.
        -   **95% CI**: Lighter, broad uncertainty band.
    -   **Artifact Generation**: Produces the high-resolution dual-band plots found in `plots/test3_uncertainty_full/`.

## 3. Data Utilities

### `check_data_size.jl`
A utility script to quickly verify the structural integrity of the loaded graph data.
-   **Checks**: Adjacency matrix reconstruction, node feature dimensions (`Variables x Time x Nodes`), and normalization ranges.
-   **Usage**: Run this before training if you suspect data corruption or dimension mismatch errors.

---

## 4. Generalization & Transfer Learning

This section covers the scripts used to validate the model's ability to generalize to unseen states (Zero-Shot).

### `Test/predict_zero_shot.jl`
-   **Purpose**: Performs Zero-Shot inference on a set of test states (e.g., West Coast) using a model trained on disjoint states (e.g., East Coast).
-   **Key Mechanisms**:
    -   **Spectral Normalization**: Detects if the test subgraph has unstable eigenvalues ($\lambda > 1$) and normalizes the adjacency matrix $A$ to ensure stability.
    -   **Latent Transfer**: Calculates statistics ($\mu, \sigma$) from the training set's latent vectors and uses them to initialize the latents for new states, avoiding "cold start" issues.
    -   **Full Horizon**: Extrapolates dynamics beyond the training window (Day 180+).

### `Test/verify_model_on_train.jl`
-   **Purpose**: A "Sanity Check" script.
-   **Usage**: Before attempting transfer learning, this script verifies that the saved checkpoint (`.jld2`) can correctly reproduce the dynamics of a state *it was trained on* (e.g., CA).
-   **Diagnosis**: If this script fails (high MSE), the checkpoint is corrupt or the model failed to converge, rendering any zero-shot attempt invalid.

---

## 5. Counterfactual Analysis & Causal Reasoning

### `Test/counterfactual_analysis.jl`
-   **Purpose**: Validate that the model has learned **causal relationships** (not just correlations) by simulating policy interventions that never occurred in the real data.
-   **Key Mechanisms**:
    -   **Dynamic Graph Switching**: The ODE solver receives different graph topologies during simulation. For lockdown scenarios, edges connecting the intervention state (NY) are severed to simulate border closure.
    -   **Covariate Manipulation**: Input splines (Indoor Events, Doctor Visits, CLI) are artificially reduced (e.g., to 10% for lockdown) while leaving the cases variable untouched - the model predicts it naturally.
    -   **Gradual Reopening**: Implements linear ramp-up of both inputs and graph connections over 30 days to avoid artificial shocks.
    -   **Multi-State Output**: Generates both detailed single-state plots and multi-state comparison grids to observe spatial effects.
-   **Key Finding**: Model demonstrates causal understanding (reduced inputs → reduced cases), but network structure contributes minimally - dynamics are dominated by local inputs and latent features.

---

## 6. Graph Topology & Ablation (Test 7b validation)

### `Resultados/test-7/fine_tune_{full,isolated,random}.jl`
-   **Purpose**: Evaluation of three competing topological hypotheses.
-   **Methodology**:
    -   Trains three identical architectures differing only in graph structure (Real, Empty, Random).
    -   Deep Fine-Tuning: 1000 epochs with `dropout=0.0`.
-   **Outcome**: Produces the checkpoints used to verify that spatial structure accelerates optimization.

### `Resultados/test-7/counterfactual_generalization_test7b.jl` (and `plot_comprehensive_zeroshot.jl`)
-   **Purpose**: The definitive Zero-Shot test (Expanded to 15 States).
-   **Key Mechanisms**:
    -   **Mean Latent Transfer**: Transfers the "average health state" of trained regions to unseen ones.
    -   **Multi-Horizon Scoring**: Calculates MSE at 20, 60, 80, 100, 200, 400 days.
-   **Key Finding**: The **Full Graph** model acts as a "Safety Net". While it ties with the Isolated model on average (due to contagion from untrained neighbors), it prevents catastrophic failures in complex states (e.g., Maryland), improving worst-case performance by **16x**.

### `Resultados/test-7/plot_comprehensive_zeroshot.jl`
-   **Purpose**: Visualization generator.
-   **Output**: Produces the multi-line comparative plots showing the divergent behaviors ("Safety Net" vs "Contagion") of the models.

---

## Documentation Navigation

For a broader context of the project, refer to:

-   **[Project Root](../README.md)**: Installation, dependencies, and high-level architecture.
-   **[Test 3 Results (Uncertainty)](../Resultados/test-3/README.md)**: Best performing model for probabilities.
-   **[Test 7 Results (Spatial)](../Resultados/test-7/README.md)**: Definitive proof of graph value and generalization.
-   **[Experiment Summary](Experiments_Summary.md)**: A comparative look at all architectures.

---

## 7. Critical Mass & Scale (Test 8)

This section details the large-scale experiment (25 Training States) designed to prove the "Critical Mass" hypothesis.

### `Resultados/test-8/train_25_states.jl`
-   **Purpose**: The main training script for the scaled-up model.
-   **Key Features**:
    -   Loads the full dataset but filters for a specific 25-state mask.
    -   Implements the standard Curriculum Learning pipeline (T2) followed by Deep Fine-Tuning.

### `Resultados/test-8/evaluate_unseen.jl`
-   **Purpose**: The central analysis script for Test 8.
-   **Outputs**:
    -   Generates the MSE tables for both Trained and Unseen states.
    -   Calculates "Peak Shift" (temporal synchronization accuracy).
    -   Produces the `plots/all_states_forecast_split.png` visualization.

### `Resultados/test-8/sanity_check.jl`
-   **Purpose**: A rigorous verification suite.
-   **Tests**:
    1.  **Random Baseline**: Confirms that untrained weights yield MSE > 200 (proving the model learns physics, not just mean).
    2.  **Isolated vs Full (Ablation)**: Runs specific states (PA, TN) in isolation vs connected mode to distinguish between "Intrinsic Resilience" (generalization via weights) and "Active Anchoring" (generalization via neighbors).

---

## 8. Graph Characterization (Test 9)

### `Resultados/test-9/graph_characterization.jl`
- **Purpose**: Topological profiling of the US states network.
- **Metrics**: Implementations for Weighted Degree, Betweenness, Eigenvector, Clustering, and Closeness Centrality.
- **Holdout Logic**: Implements a constrained stratified sampling algorithm with an auto-repair loop based on Fiedler values.

### `Resultados/test-9/umap_analysis.jl`
- **Purpose**: Exploratory nonlinear manifold analysis.
- **Mechanism**: Sweeps through `n_neighbors` and `min_dist` parameters in UMAP to verify cluster stability and visualize the 5D topological space in 2D.

