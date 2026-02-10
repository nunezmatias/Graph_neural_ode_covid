# Experiment: Test 4 (No Covariates)

## Hypothesis
Can the Graph Neural ODE Learn the dynamics of COVID-19 spread using **only** the history of active cases and static latent variables, **without** any external covariates (mobility, weather, mask usage)?

## Configuration
-   **Model**: GNN-ODE (Latent Dimension = 3)
-   **Inputs**: Active Cases (1 channel) + Latent Features (3 channels)
-   **Covariates**: None (Removed)
-   **Architecture**: `4 -> 32 -> 32 -> 32 -> 1` (Same width as Test 2)
-   **Training**: Curriculum Learning (5 -> 180 points), AdamW, Patience=50.

## Results
**Status:** FAILED / DIVERGED

The model training was stopped at **Stage 180** (epoch ~920) due to massive instability.

-   **Stage 10-60**: The model maintained acceptable loss values (~0.2 - 50.0).
-   **Stage 90**: Loss increased significantly to >500.
-   **Stage 120**: Loss stabilized around 78.0 after reducing LR.
-   **Stage 180 (CRITICAL FAILURE)**: The loss exploded to **62,600+**.

### Conclusion
**The hypothesis is rejected.**
The Graph Neural ODE **cannot** learn the long-term dynamics (180 days) using only minimal latent variables and infection history. The absence of external covariates (Community Mobility, Weather, Events) makes the optimization landscape impossible to traverse for longer time horizons. The model "forgets" or diverges when forced to extrapolate beyond 120 days.

## Visualizations
-   `plots/loss_divergence.png`: Shows the catastrophic failure at Stage 180.

## Reproducibility

To reproduce this failed experiment (Ablation Study):

1.  Modify `Train/model_opt.jl`:
    *   Set `nin_covar = 0` (Line 31).
    *   In `dudt` function (Line 179 & 218), comment out `cov_matrix` and remove it from `vcat`:
        ```julia
        # cov_matrix = map(s -> s(t), splines)
        model_input = vcat(u_reshaped, latents) # Removed cov_matrix
        ```
2.  Run the training:
    ```bash
    julia --project=. Train/model_opt.jl
    ```

### Running From Scratch
This command inherently attempts to train the model from scratch. Since this experiment is expected to fail/diverge, "running from scratch" simply means observing the loss explosion during this training process.
