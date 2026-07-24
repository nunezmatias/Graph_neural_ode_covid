# Graph Neural Network Training Pipeline

This folder contains Julia scripts designed for training Graph Neural Network (GNN) models on time-series data. It includes two distinct architectural approaches and a 5-phase training pipeline to ensure optimal convergence.

---

## Repository Structure

* **`training_GCN.jl`**: Contains the model architecture and training loops using Julia's built-in `GCNConv` layer.
* **`training_custom.jl`**: Contains the model architecture and training loops utilizing a custom, hand-crafted message passing layer. Bias is shared across nodes.
* **`training_custom_V2.jl`**: Contains the model architecture and training loops utilizing a custom, hand-crafted message passing layer. Bias is node-specific.
* **`training_custom_V3.jl`**: Contains the model architecture and training loops utilizing a custom, hand-crafted layer adding two separate terms: W_self (the node dynamic optimization term) and W_neigh (the diffusion across the graph term). Bias is shared across nodes. Minimum ratio of spatial-to-local weights (||W_neigh|| / ||W_self||) is controlled by the parameter reg_ratio, reg_lambda is the Penalty strength multiplier added to the loss function to enforce the target minimum ratio.
* **`run_training.jl`**: The main execution script. It loads the dataset, builds the necessary adjacency matrix, and initiates the training process. For the matrix normalization function you can choose the weights to assign to self-loops (0 to 1).

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

---

## How to Run

The primary entry point for the pipeline is `run_training.jl`. 

### 1. Select Your Architecture
Inside `run_training.jl`, there are two specific code blocks for defining the adjacency matrix and calling the training function—one tailored for the **GCN layer** and one for the **Custom layer**. 
> **Note:** You must manually comment out the architecture you do not wish to use and uncomment the one you need before running the script.

### 2. Execute the Script
You can specify the directory where the optimized parameters and loss history will be saved by passing it as a command-line argument.

Run the following command in your terminal:

```bash
julia run_training.jl <save_path>
```

---

## Challenges

Training with Julia’s built-in GCNConv layer is computationally slow. To address this, I implemented a custom diffusion layer that significantly improves training speed while preserving the desired diffusion behavior.

The main challenge is that the predicted dynamics tend to be highly correlated across nodes. In practice, the model often learns a common “average” trajectory and then scales it differently for each node, rather than capturing truly node-specific dynamics. I investigated the effect of the self-loop weight and found that increasing it leads to more heterogeneous, node-specific predictions. However, the diffusion mechanism becomes weaker and the model loses its ability to effectively propagate information between nodes.

The custom implementation makes use of a shared bias term across all nodes. To increase node-level flexibility, I modified the layer so that each node has its own bias parameters ("custom_V2"). Preliminary results suggest that this approach can achieve a better balance between node-specific behavior and diffusion, although the overall diffusion strength remains relatively weak and needs tuning of the self-loop weights. Work in progress.
