# Future Experiments: Extending the "Critical Mass" Hypothesis

This document outlines a series of follow-up experiments designed to further explore, stress-test, and extend the findings from **Test 8: The Critical Mass Experiment**. Test 8 demonstrated that training on 50% of the US states (25 nodes) is sufficient to stabilize predictions for the remaining 50%, provided they are topologically connected to the training cluster.

The experiments below are grouped into three categories: **Optimization**, **Robustness**, and **Generalization**.

---

## Part I: Optimization — Finding the Minimum Viable Network

These experiments aim to reduce the training data requirements while maintaining (or improving) performance.

### 1. The "Strategic Sentinel" Experiment

**Core Question:** *Can we achieve Critical Mass with fewer states by choosing them strategically?*

The current experiment used 25 states selected based on data availability and geographic diversity. However, graph theory tells us that not all nodes are created equal. Some states (e.g., Texas, Illinois) act as "hubs" connecting multiple regions, while others (e.g., Maine, Wyoming) are peripheral.

**Hypothesis:** A training set of 15 high-centrality states will outperform a training set of 25 randomly selected states.

**Methodology:**
1.  Compute the **Betweenness Centrality** and **PageRank** scores for all 50 states using the mobility adjacency matrix.
2.  Select the top-K states by centrality as the "Strategic" training set.
3.  Train three models:
    *   **Strategic-15:** Top 15 by centrality.
    *   **Random-15:** Random selection of 15 states.
    *   **Peripheral-15:** Bottom 15 by centrality.
4.  Evaluate all three on the same 35 unseen states.

**Expected Outcome:** The Strategic model should match or exceed the performance of Random-25 (the original Test 8), proving that **network influence is more valuable than sample size**.

**Implications:** If successful, this experiment would provide a principled method for selecting sentinel locations in future pandemic surveillance systems.

---

### 2. The Inverse Ablation: Finding the "Collapse Point"

**Core Question:** *What is the minimum percentage of coverage needed before the model fails catastrophically?*

Test 8 showed that 50% coverage works. But where is the threshold? Is 30% enough? 20%? There must be a "phase transition" point where the GNN stops learning physics and reverts to mean-prediction.

**Hypothesis:** There exists a critical threshold $\theta_c$ (likely between 15% and 30%) below which the Test MSE increases exponentially.

**Methodology:**
1.  Train a series of models with 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50% of states.
2.  Plot the Test MSE (on all unseen states) as a function of coverage percentage.
3.  Identify the "elbow" in the curve — the point where adding more states yields diminishing returns.

**Expected Outcome:** A clear S-curve or exponential decay, with a critical point around 20-25%.

**Implications:** This defines the **Minimum Viable Network** for future deployments, allowing public health agencies to know the smallest sensor network needed for reliable forecasting.

---

## Part II: Robustness — Testing Limits Under Stress

These experiments probe the model's resilience to distributional shifts, adversarial conditions, and missing data.

### 3. Temporal Critical Mass: The Time-Space Trade-off

**Core Question:** *If we have less time-series data, can we compensate by training on more states (and vice versa)?*

Test 8 used 180 days of training data. In a real-world scenario (the start of a new pandemic), we might only have 30-60 days. Can spatial density compensate for temporal scarcity?

**Hypothesis:** There exists a quantifiable exchange rate between time and space — for example, "1 extra month of data is equivalent to 5 extra states."

**Methodology:**
1.  Train a matrix of models with:
    *   **Temporal Axis:** 60, 90, 120, 180 days.
    *   **Spatial Axis:** 10, 20, 30, 40 states.
2.  Evaluate all models on the full 400-day horizon for the unseen states.
3.  Plot **iso-performance contours** (lines of equal MSE) on a Time x Space heatmap.

**Expected Outcome:** A hyperbolic curve showing that extreme deficits in one dimension can be offset by surpluses in the other, up to a limit.

**Implications:** This provides a practical guide for early-pandemic decision-making: "We only have 45 days of data — how many sentinel sites do we need?"

---

### 4. The "Variant" Stress Test: Distribution Shift

**Core Question:** *Can a model trained on the original COVID-19 strain adapt to Delta or Omicron with minimal fine-tuning?*

The physics of viral spread (diffusion through social networks) should be invariant across variants. What changes are the epidemiological parameters ($R_0$, incubation period). If the GNN learned true physics, it should adapt quickly.

**Hypothesis:** A Critical Mass model can achieve competitive performance on a new variant with only 14 days of fine-tuning data.

**Methodology:**
1.  Take the model trained on Waves 1-2 (Days 0-400, Wild Type).
2.  **Freeze** all GNN layers (spatial logic).
3.  Fine-tune **only** the ODE parameters ($\beta$, $\gamma$) using the first 14 days of the Delta Wave (if data is available).
4.  Evaluate on the remaining Delta Wave timeline.

**Expected Outcome:** The fine-tuned model should recover performance faster than training from scratch.

**Implications:** This establishes the model as a **Few-Shot Learning** framework for pandemic response.

---

### 5. Rural vs. Urban: Sociodemographic Heterogeneity

**Core Question:** *Does the model overfit to metropolitan dynamics?*

Most of the training data comes from highly populated states (NY, CA, TX, FL). The "Topological Isolates" that fail (WY, SD, ND) are also among the least populous. Is it topology or demography that causes the failure?

**Hypothesis:** A model trained exclusively on urban states will fail on rural states *even if they are well-connected*, because the transmission dynamics are fundamentally different.

**Methodology:**
1.  **Experiment A (Urban → Rural):**
    *   Train Set: Top 20 states by population.
    *   Test Set: Bottom 20 states by population.
2.  **Experiment B (Rural → Urban):**
    *   Train Set: Bottom 20 states by population.
    *   Test Set: Top 20 states by population.
3.  Compare asymmetry in performance.

**Expected Outcome:** Urban → Rural transfer will fail more severely than Rural → Urban, because urban dynamics are "simpler" (faster mixing) while rural dynamics are more heterogeneous.

**Implications:** This would motivate a **stratified sampling** strategy for future training sets.

---

### 6. Adversarial Graph Attack: Noise Robustness

**Core Question:** *How sensitive is the model to errors in the mobility graph?*

In practice, mobility matrices are estimates from cell phone data, which are noisy. What happens if we corrupt the graph with spurious edges (e.g., connecting Hawaii to Montana)?

**Hypothesis:** The Critical Mass model has some inherent robustness to noise, but will degrade significantly if more than 15% of edges are corrupted.

**Methodology:**
1.  Take the trained Critical Mass model.
2.  Add 5%, 10%, 15%, 20% random edges to the adjacency matrix (connecting previously unconnected states).
3.  Evaluate performance degradation.
4.  Compare with the Isolated Model (baseline with no topology), which should be immune to graph noise.

**Expected Outcome:** A graceful degradation curve, with significant drops after 15% corruption.

**Implications:** This quantifies the **data quality requirements** for the mobility graph.

---

## Part III: Generalization — Pushing Beyond COVID-19 and the US

These experiments test whether the learned physics can transfer to entirely new contexts.

### 7. Cross-Country Transfer: Europe

**Core Question:** *Can a model trained on US states predict dynamics in European countries?*

If the GNN has learned universal epidemiological physics (the interplay of mobility, seasonality, and social behavior), it should transfer to any population graph with similar structure.

**Hypothesis:** A US-trained Critical Mass model can achieve reasonable Zero-Shot performance on a European graph (e.g., EU member states) if we normalize the covariates appropriately.

**Methodology:**
1.  Obtain a mobility matrix for Europe (e.g., from Google Mobility Reports or Eurostat).
2.  Normalize inputs (COVID cases, mobility indices) to the same scale as the US training data.
3.  Apply the frozen US model to predict European dynamics.
4.  Fine-tune for 100 epochs and measure improvement.

**Expected Outcome:** Initial Zero-Shot will be poor (due to scale/structure mismatch), but fine-tuning will converge quickly, leveraging the pre-learned spatial logic.

**Implications:** This establishes**Pandemic Transfer Learning** as a viable paradigm.

---

### 8. Granularity Scaling: County-Level Modeling

**Core Question:** *Does Critical Mass work at finer spatial resolutions?*

The US has ~3,100 counties vs. 50 states — a 60x increase in resolution. Does the 50% coverage rule still hold, or does local redundancy reduce the required fraction?

**Hypothesis:** At county resolution, Critical Mass will be achieved with a *smaller* fraction of nodes (e.g., 20%) because of increased local redundancy.

**Methodology:**
1.  Build a county-level mobility graph (using CDC or SafeGraph data).
2.  Train models with 5%, 10%, 20%, 30% of counties.
3.  Compare the Coverage vs. MSE curve with the state-level curve.

**Expected Outcome:** The "collapse point" will shift leftward, requiring fewer nodes proportionally.

**Implications:** This supports **Scale-Free** deployment of the model at arbitrary administrative resolutions.

---

### 9. Sequential Wave Transfer: Temporal Generalization

**Core Question:** *Can a model trained on Wave 1 predict Wave 2 without retraining?*

Waves differ in timing, amplitude, and variant. But the underlying network structure and human behavior patterns remain similar.

**Hypothesis:** A model trained on Days 0-150 (First Wave) will predict days 200-400 (Second Wave) with lower error than a naive baseline, proving temporal generalization.

**Methodology:**
1.  Train exclusively on Days 0-150.
2.  Evaluate Zero-Shot on Days 200-400 (completely unseen).
3.  Compare with: (a) a model trained on all data, and (b) a naive "repeat last value" baseline.

**Expected Outcome:** The Zero-Shot model will capture the *timing* of Wave 2 peaks (from network propagation) but miss the exact *amplitude* (due to unobserved variant effects).

**Implications:** This validates **Early Warning System** potential using partial data.

---

### 10. Reverse-Time ODE: Causal Validation

**Core Question:** *Can the model reconstruct the past from the present?*

A truly causal model should satisfy time-reversal symmetry. If we know the state at Day 180, can we integrate the ODE backward to recover Day 0?

**Hypothesis:** A well-trained Neural ODE will approximately recover the initial conditions, proving it has learned reversible physics (not just curve-fitting).

**Methodology:**
1.  Take the final state at Day 180.
2.  Integrate the Neural ODE with $dt < 0$ (reverse time).
3.  Compare the reconstructed Day 0 state with the actual historical data.

**Expected Outcome:** Partial success — the model will recover the general shape but not the exact values (due to accumulated numerical error and chaotic sensitivity).

**Implications:** This is a novel **Causal Validation** technique for Neural ODEs.

---

### 11. Bridges and Islands: Topological Vulnerability

**Core Question:** *Which states, if removed, would cause the network to collapse?*

Some states act as "bridges" connecting otherwise disconnected regions. If Texas is removed, the East and West coasts lose their main land route.

**Hypothesis:** Removing "bridge" states from training will cause catastrophic failure in the regions they connect, even if those regions have other trained nodes.

**Methodology:**
1.  Identify **Cut Vertices** (articulation points) in the mobility graph.
2.  Train a model *excluding* the top 3 cut vertices.
3.  Evaluate performance on states that were connected *through* those bridges.

**Expected Outcome:** The "Safety Net" effect will collapse for the bridged regions, with MSE spikes similar to the Topological Isolates in Test 8.

**Implications:** This identifies **Critical Infrastructure Nodes** for pandemic surveillance.

---

### 12. Bayesian Calibration: Uncertainty vs. Density

**Core Question:** *Does higher training density improve uncertainty calibration?*

Test 3 introduced MC Dropout for uncertainty. But are those confidence intervals *calibrated*? Does training on more states make them more reliable?

**Hypothesis:** A Critical Mass model will have better-calibrated uncertainty bands (closer to 95% empirical coverage) than a sparse model.

**Methodology:**
1.  Train models with 10, 25, 40 states.
2.  Generate 95% CI bands using MC Dropout (100 samples).
3.  Calculate **Coverage Probability**: the fraction of true observations falling within the 95% CI.
4.  A well-calibrated model should achieve ~95% coverage.

**Expected Outcome:** The 10-state model will be overconfident (< 80% coverage), while the 40-state model will approach 95%.

**Implications:** This establishes a link between **Sample Size and Bayesian Reliability**.

---

## Summary Table

| # | Experiment Name | Category | Primary Question |
|---|-----------------|----------|------------------|
| 1 | Strategic Sentinel | Optimization | Centrality vs. Random Selection |
| 2 | Collapse Point | Optimization | Minimum Coverage Threshold |
| 3 | Time-Space Trade-off | Robustness | Temporal vs. Spatial Compensation |
| 4 | Variant Stress Test | Robustness | Few-Shot Adaptation to New Strains |
| 5 | Rural vs. Urban | Robustness | Sociodemographic Overfitting |
| 6 | Adversarial Attack | Robustness | Graph Noise Sensitivity |
| 7 | Cross-Country Transfer | Generalization | US → Europe |
| 8 | County-Level Scaling | Generalization | Resolution Invariance |
| 9 | Sequential Wave Transfer | Generalization | Wave 1 → Wave 2 |
| 10 | Reverse-Time ODE | Generalization | Causal Reversibility |
| 11 | Bridges and Islands | Generalization | Topological Vulnerability |
| 12 | Bayesian Calibration | Generalization | Uncertainty vs. Density |

---

## Prioritization Recommendation

For a PhD thesis or publication, I recommend the following order based on novelty and feasibility:

1.  **Strategic Sentinel (1)** — High impact, easy to implement.
2.  **Collapse Point (2)** — Foundational for all other claims.
3.  **Time-Space Trade-off (3)** — Directly actionable for public health.
4.  **Variant Stress Test (4)** — High novelty, requires external data.
5.  **Cross-Country Transfer (7)** — Most ambitious, best for a "grand finale."
