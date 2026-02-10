# Experiment: Test 9 — Graph Characterization & Stratified Holdout Selection

## Hypothesis
Can we systematically select a subset of 9 US states for holdout validation that is **topologically representative** of the full gravity graph, while preserving the structural integrity of the 40-state training graph?

## Methodology

### Phase 1: Load Gravity Graph
- Load the pre-computed adjacency matrix from `Data/adj_pop_dist.csv` (49 continental states + DC)
- Build a weighted graph (Graphs.jl), remove self-loops
- The weights represent gravity-model coupling: $A_{ij} = P_i^{0.5} P_j^{0.5} \exp(-d_{ij}/500)$
- **Key finding:** The graph is **fully connected** (1176 edges = 49×48/2) — every pair of states has nonzero weight

### Phase 2: Topological Feature Extraction (5D Vector per State)
For each node, compute:

| Metric | Rationale |
|---|---|
| **Weighted Degree** | Total gravity flow — identifies "Super-Spreaders" |
| **Betweenness Centrality** | Flow control — identifies structural "Bridges" |
| **Eigenvector Centrality** | Connection quality — linked to NY (high risk) vs linked to ND (low risk) |
| **Clustering Coefficient** | Redundancy / "Safety Net" — can the model survive without this node? |
| **Closeness Centrality** | How "nearby" a node is in weighted-path distance (replaced k-Core, which was constant=48 on the fully connected graph) |

### Phase 3: Embedding & Clustering
1. **StandardScaler** (Z-score) normalization
2. **PCA** reduction to 2D for visualization (PC1=59.9%, PC2=19.2%, Total=79.1%)
3. **UMAP** with multiple parameter sweeps for nonlinear embedding comparison
4. **KMeans** (k=3) clustering → semantically labeled as:
   - **Cluster A (Hubs):** High degree/eigenvector, low clustering
   - **Cluster B (Connectors):** Moderate degree, high clustering
   - **Cluster C (Periphery):** Low degree, low clustering, bridge role via betweenness

### Phase 4: Stratified Holdout Sampling
- 3 states from Cluster A + 3 from B + 3 from C = 9 holdout states
- **Geographic constraint:** No more than 4 states from the same US Census Region (Northeast, Midwest, South, West)

### Phase 5: Integrity Check ("Lobotomy Check")
- Remove the 9 holdout nodes from the graph → subgraph G'
- Verify G' is **connected** and the **Fiedler value** (algebraic connectivity) is above threshold
- **Auto-repair:** If a bridge node is in the holdout, swap it back and replace with a lower-betweenness node from the same cluster

---

## Results

### Topological Metrics Summary

| Metric | Min | Max | Observation |
|---|---|---|---|
| Weighted Degree | 2.4M (MT) | 51.0M (PA) | Population-driven |
| Betweenness | 0.0 (24 states) | 0.384 (IL) | Only ~25 states are on shortest paths |
| Eigenvector | 0.0001 (WA) | 0.068 (PA) | Strongly correlated with degree |
| Clustering | 0.228 (CA) | 0.918 (ND) | Inversely correlated with degree |
| Closeness | 148K (MT) | 652K (IL) | Geographic centrality |

### Cluster Composition

| Cluster | N | Profile | Members |
|---|---|---|---|
| **A_Hubs** | 20 | High degree, high eigenvector, low clustering | FL, IL, MD, NC, CT, NJ, PA, GA, AL, OH, TX, SC, TN, KY, NY, MI, MO, IN, MA, VA |
| **B_Connectors** | 22 | Moderate degree, high clustering | WV, MN, RI, NH, VT, DE, NM, WI, NE, LA, CO, OK, WY, ND, ME, AR, MS, MT, KS, SD, DC, IA |
| **C_Periphery** | 7 | Low degree, low clustering, high betweenness (bridge role) | ID, CA, OR, WA, UT, NV, AZ |

### PCA Visualization

![PCA Clusters](pca_clusters.png)

### Final 40/9 Split

**Holdout (9):** `AZ, LA, MA, MD, NM, NV, RI, TN, UT`

| Cluster | Holdout States |
|---|---|
| A_Hubs | MA, MD, TN |
| B_Connectors | LA, NM, RI |
| C_Periphery | AZ, NV, UT |

**Region distribution:** West (4), South (3), Northeast (2) ✓

### Integrity Check
- Training graph (40 states): **Connected** ✓
- Fiedler value: **767,735** (no repair needed)

### UMAP Analysis

UMAP embeddings with multiple parameter sets to explore nonlinear structure:

![UMAP Parameter Sweep](umap_sweep.png)

---

## Outputs

| File | Description |
|---|---|
| `metrics_table.csv` | Full 49-state topological metrics |
| `pca_clusters.png` | PCA scatter plot with clusters and holdout marked |
| `umap_sweep.png` | UMAP embeddings with multiple n_neighbors / min_dist |
| `holdout_selection.json` | Final training (40) and holdout (9) sets |

## Reproducibility

```bash
cd /path/to/Graph_neural_ode_covid
# Main analysis (PCA + clustering + holdout selection)
julia --project=. Resultados/test-9/graph_characterization.jl

# UMAP visualization sweep
julia --project=. Resultados/test-9/umap_analysis.jl
```

**Requirements (Project.toml):** `Graphs`, `SimpleWeightedGraphs`, `CSV`, `DataFrames`, `Plots`, `JSON`, `LinearAlgebra`, `Statistics`, `UMAP`.
