# Experiment Series 3: Uncertainty Quantification (MC Dropout)

## 1. The Concept: Multiverse Generator
Until now, our Neural ODE model was a **Deterministic Oracle**: for a given input, it always provided the exact same prediction. This is dangerous in epidemiology, where "noise" and unknown variables are enormous.

In **Test-3**, we transform the model into a **Stochastic System** using the **Monte Carlo Dropout** technique.

### How does it work?
Imagine we train not just a single neural network, but a "swarm" of possible networks. We achieve this by inserting `Dropout(0.2)` layers that randomly turn off 20% of the neurons at each step.

*   **During Training:** The network learns to be redundant. It cannot rely on any individual neuron, so it distributes knowledge.
*   **During Prediction (Inference):** We do not ask for a single answer. We keep Dropout enabled and ask for 100 consecutive predictions. Each time, a slightly different sub-network gives its opinion.
    *   *Result:* We obtain 100 possible trajectories ("The Multiverse").
    *   *Consensus:* If the 100 trajectories are almost identical, the model is sure. If they diverge (some go up, others go down), the model confesses its ignorance.

---

## 2. Technical Methodology

### Architecture (Modified GNN)
Three Dropout layers were injected into the processing chain:
```julia
gnn = GNNChain(
    GraphConv(nin => 32, tanh),
    Dropout(0.2),  # <--- Chaos
    GraphConv(32 => 32, tanh),
    Dropout(0.2),  # <--- Chaos
    GraphConv(32 => 32, tanh),
    Dropout(0.2),  # <--- Chaos
    GraphConv(32 => 1)
)
```

### Inference Protocol (Sampling)
To generate epistemic uncertainty:
1.  **Activation:** Training mode is forced (`Flux.trainmode!(model, true)`).
2.  **Sampling:** The ODE is integrated $N=100$ times.
3.  **Statistics:**
    *   **Mathematical Expectation ($\mu$):** Average of the 100 simulations.
    *   **Uncertainty ($\sigma$):** Standard deviation step by step.
    *   **Confidence Interval:** Band of $\mu \pm 1.96\sigma$ (95% CI).

---

## 3. Expected Visualization

We will generate two types of plots to diagnose uncertainty:

1.  **Spaghetti Plot:** Raw visualization of the 100 overlapping lines. Useful for detecting bimodality (does the model believe it can go UP or DOWN, but nothing in between?).
2.  **Confidence Band Plot:** Clean plot with a "shadow" around the mean line. This is the final report chart.

---

## 4. Practical Guide: Execution and Experimentation

This section explains how to run the codes and modify parameters to perform new experiments.

### 4.1 Running the Uncertainty Analysis
To generate the band plots (95% and 50%) using the already trained model:

```bash
# From the repository root
julia --project=. Test/uncertainty_analysis_full.jl
```
*Output:* Will generate `*_forecast_dual.png` images in `plots/test3_uncertainty_full/`.

### 4.2 Experiment: Changing the Number of Samples
If you want greater statistical precision (smoother bands) or higher speed:

1.  Open `Test/uncertainty_analysis_full.jl`.
2.  Search for line `159`:
    ```julia
    N_SAMPLES = 100
    ```
3.  **Modify:**
    *   `N_SAMPLES = 1000` -> Ultra-smooth bands (Slow, ~10 mins).
    *   `N_SAMPLES = 20` -> Quick test (Draft, ~30 secs).

### 4.3 Experiment: Changing the Uncertainty Level (Dropout Rate)
**IMPORTANT:** Dropout is part of the trained structure. To change it, **you must retrain the model**.

1.  **Step 1: Modify Training**
    *   Open `Train/model_opt.jl`.
    *   Search for line `150` (Model Definition):
        ```julia
        model = ExplicitGNN(nin_tot, 32, nout, 0.2) # Change 0.05 to 0.2
        ```
    *   A value of `0.2` will generate much wider bands (higher uncertainty). `0.01` will make them almost invisible.

2.  **Step 2: Retrain**
    ```bash
    julia --project=. Train/model_opt.jl
    ```
    This will save new weights to `Params/par_opt_test3.jld2`.

3.  **Step 3: Update Analysis**
    *   Open `Test/uncertainty_analysis_full.jl`.
    *   Make sure to also change the dropout in the model loading (Line 150) to match the training:
        ```julia
        model = ExplicitGNN(nin_tot, 32, nout, 0.2) # Must match Train!
        ```
    *   Run the analysis script again.

### 4.4 Experiment: Visualizing Different Confidence Intervals
By default, we show 95% and 50%. To see extreme risk (e.g., 99%):

1.  Open `Test/uncertainty_analysis_full.jl` (Lines ~205).
2.  Modify the quantiles:
    ```julia
    # For 99% confidence (Covers almost everything)
    pred_lower_99 = quantile(x, 0.005)
    pred_upper_99 = quantile(x, 0.995)
    ```

**Final Goal:** Deliver not just a prediction, but a measure of *confidence*. Knowing when the model "doesn't know" is as important as being right.
