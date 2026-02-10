# Test 6: Counterfactual Analysis (Causal Reasoning)

## Objective
Demonstrate that the GNN-ODE model has learned **causal relationships** between mobility/activity inputs and COVID-19 transmission, not just statistical correlations. We test this by simulating policy interventions (lockdowns) that never occurred in the real data.

## Methodology

### Intervention Design
We simulate three scenarios for New York (NY):

1. **Baseline**: Historical inputs (no intervention)
2. **Lockdown + Isolation**: 
   - **Covariate Reduction**: Indoor Events, Doctor Visits, and CLI reduced to 10% (Day 60-120)
   - **Graph Topology Change**: All edges connecting NY to other states are severed (simulating border closure)
   - **Gradual Reopening**: Linear ramp-up over 30 days (Day 120-150)
3. **Permanent Reduction**: 20% reduction in all inputs from Day 60 onwards (no reopening)

### Key Implementation Details
- **What we modify**: Only the **input covariates** (Indoor Events, Doctor Visits, CLI) and **graph edges**
- **What we DON'T touch**: The **cases variable** - the model predicts this naturally based on modified inputs
- **Dynamic Graph**: The ODE solver receives `g_lockdown` (isolated NY) during Day 60-120, then switches back to `g_normal`
- **Units**: All predictions are in log-transformed case counts

## Results

### 1. NY Direct Effects
![NY Counterfactual](plots/NY_counterfactual_final.png)

**Observations:**
- **Green Line (Lockdown)**: Cases drop dramatically during intervention (Day 60-120), nearly reaching zero
- **Rebound Effect**: After reopening (Day 150+), cases return to baseline trajectory
- **Red Line (Permanent)**: Sustained reduction throughout, demonstrating long-term policy impact

**Interpretation:**
The "snap-back" to baseline is **correct behavior** for a forcing-driven system:
- The model learned that transmission is dominated by **external inputs** (activity levels)
- There is no significant **susceptible depletion** (population >> infected)
- When inputs return to 100%, the system returns to its natural attractor
- This validates the model understands **causality**: reduced activity → reduced transmission

### 2. Spatial Effects (Multi-State Analysis)
![Multi-State Comparison](plots/multi_state_comparison.png)

**Critical Finding:**
NY's lockdown has **minimal to no effect** on neighboring states (FL, IL, NC, NJ).

**Implications:**
1. **Local Dominance**: The model learned that each state's dynamics are primarily driven by **local inputs**, not spatial diffusion
2. **Weak Network Effects**: Graph edges contribute little to predictions - latent features and local covariates dominate
3. **Possible Explanations**:
   - Real-world data showed weak inter-state coupling during this period
   - State-specific latents capture most variability
   - GNN aggregation (`mean`) dilutes neighbor signals

## Key Learnings

✅ **Causal Understanding Validated**: Model correctly predicts case reduction from input reduction alone

✅ **Forcing-Driven Dynamics**: Transmission is dominated by activity levels, not immunity depletion

⚠️ **Limited Spatial Coupling**: Network structure has minimal impact on predictions - model is primarily local

❓ **Future Work**: Ablation study comparing graph-based vs. isolated-node training to quantify network contribution

## Technical Notes

**Data Verification:**
- Raw cases (Day 1, NY): 1,585 → Log: 7.37
- Raw cases (Day 200, NY): 7,874 → Log: 8.97

**Neighbors Identified:**
FL, IL, NC, NJ, GA, OH, VA (7 states with non-zero edges to NY)

**Scripts:**
- `Test/counterfactual_analysis.jl`: Main simulation
- Output: `plots/NY_counterfactual_final.png`, `plots/multi_state_comparison.png`

## Reproducibility

To run the Counterfactual Analysis and generate the scenarios:

```bash
julia --project=. Test/counterfactual_analysis.jl
```

This script will:
1.  Load the trained model parameters (`Params/par_opt_test3.jld2`).
2.  Simulate the three scenarios (Baseline, Lockdown, Permanent Reduction).
3.  Generate the comparative plots in `Resultados/test-6/plots/`.

### Running From Scratch (No pre-trained weights)
This analysis depends on the model from **Test 3**. If `Params/par_opt_test3.jld2` is missing:

1.  **Train the Base Model (Test 3):**
    ```bash
    # Ensure Train/model_opt.jl is configured for Test 3 (Latent 3, Dropout enabled)
    julia --project=. Train/model_opt.jl
    ```
2.  **Run the Counterfactuals:**
    ```bash
    julia --project=. Test/counterfactual_analysis.jl
    ```
