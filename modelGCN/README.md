# Graph Neural Network Training Pipeline

This folder contains Julia scripts designed for training Graph Neural Network (GNN) models on time-series data. It includes two distinct architectural approaches and a 5-phase training pipeline to ensure optimal convergence.

---

## Repository Structure

* **`training_GCN.jl`**: Contains the model architecture and training loops using Julia's built-in `GCNConv` layer.
* **`training_custom.jl`**: Contains the model architecture and training loops utilizing a custom, hand-crafted message passing layer. Bias is shared across nodes.
* **`training_custom_V2.jl`**: Contains the model architecture and training loops utilizing a custom, hand-crafted message passing layer. Bias is node-specific.
* **`run_training.jl`**: The main execution script. It loads the dataset, builds the necessary adjacency matrix, and initiates the training process.
* **`run_training_V2.jl`**: The main execution script. Adds a new normalization function for the adjacency matrix where you can choose the weights to assign to self-loops (0 to 1).

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
