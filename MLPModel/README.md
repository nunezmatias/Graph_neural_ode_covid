# Graph Neural ODE: architecture with MLP

This folder contains Julia scripts designed for training Graph Neural ODE models on time-series data. 

---

## Model architecture

The main architecture is made of 4 layers: a dense MLP layer, two message passing layers and one MLP layer in output. Number of hidden channels is 128.

---

## Repository Structure

* **`Training`**: the folder contains the 5 stages training pipeline (training_mlp.jl) and the HPC launcher scripts to perform ensemble training for the temporal split (t<220 training, t>220 testing) and zero-shot approach (37 nodes in training, 12 nodes for testing), respectively in run_training_mlp_time.jl and run_training_mlp_zeroshot.jl.
* **`Parameters`**: Contains optimized parameters from the temporal and zero-shot trainings.
* **`Plots`**: Directory for saved evaluation plots.
* **`Tables`**: Contains errors for each state for the 2 trained models in a tabular format.
* **`evaluate_mlp_(temporal/zeroshot).jl`**: Evaluation scripts to generate predictions and errors tables for the two trained models.
* **`lockdown_mlp_(temporal/zeroshot).jl`**: Scripts to simulate lockdown scenarios for the two trained models.
* **`analyze_errors.ipynb`**: Preliminary notebook to analyze models errors.

---

## Errors Metrics

* **`MAE_norm_(train/test)_mean`**: Mean Absolute Error, measures point prediction error on the transformed/normalized scale ($X_{\text{norm}}$) across training ($t \le 220$) and testing ($t > 220$) horizons, reported ensemble mean with 2.5% and 97.5% ensemble quantiles.
* **`wMAPE_(train/test)_mean`**: Weighted Mean Absolute Percentage Error, measures scale-invariant error on actual case counts. Evaluated for training and full test horizons, reported ensemble mean with 2.5% and 97.5% ensemble quantiles.
* **`PerCapita100k_test_mean`**: Calculates the average daily absolute error in case numbers scaled per 100,000 residents ($\frac{\text{MAE}}{\text{population}} \times 100{,}000$), reported ensemble mean with 2.5% and 97.5% ensemble quantiles.
* **`PeakShift_test_mean`**: Quantifies temporal timing error in days by taking the index difference between predicted peak day and actual peak day ($\text{argmax}(\hat{y}) - \text{argmax}(y)$). Positive values indicate a delayed peak prediction, while negative values indicate an early prediction, reported ensemble mean with 2.5% and 97.5% ensemble quantiles.
* **`PeakAmp_ratio_test_mean`**: Measures relative magnitude error at the epidemic peak ($\frac{\max(\hat{y}) - \max(y)}{\max(y)}$). It indicates the percentage by which the model overshoots (positive) or undershoots (negative) the maximum outbreak height during the test horizon, reported ensemble mean with 2.5% and 97.5% ensemble quantiles.
* **`Coverage_(95/50)_Test`**: Measures model calibration by calculating the percentage of actual case data points that lie strictly within your predicted 50% and 95% interval boundaries. Ideal values match the target bounds (0.50 and 0.95).
* **`WIS_(Train/Test)`**: Weighted Interval Score, evaluates overall probabilistic forecast quality. It combines the absolute error of the median prediction with penalty terms for true values falling outside the predicted 50% and 95% interval bands, rewarding tight bounds while penalizing under- or over-estimation. (Actually suggested by Gemini)

---

## Training Pipeline

Regardless of the chosen architecture, both scripts employ a 5-phase sequential training strategy to escape local minima and reach an optimal global solution:

1.  **Phase 1: Temporal Chunking** * Trains on random windows of varying lengths sampled from the original time series. 
    * *Dropout:* Enabled.
2.  **Phase 2: Global Splicing** * Expands training to encompass the entire, continuous time series. 
    * *Dropout:* Enabled.
3.  **Phase 3: Stochastic Basin Hunting** * Introduces a cosine annealing learning rate scheduler to improve minimum search across the full time series. 
    * *Dropout:* Enabled.
4.  **Phase 4: Deterministic Descent** * Maintains the cosine annealing scheduler for minimum search but removes stochasticity.
    * *Dropout:* Disabled.
5.  **Phase 5: Final Convergence** * A prolonged, final training phase on the full time series using a standard learning rate to reach the asymptotic floor.
    * *Dropout:* Disabled.