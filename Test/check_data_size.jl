using JLD2, GNNLux, Lux, Graphs, CSV, DataFrames, JSON

# Load full graph logic (simplified from preprocessing)
push!(LOAD_PATH, "./Utils/GraphCreator/");
using GNNGraph_mod;

# Reconstruct Adjacency for context
A_pop = CSV.read("./Data/adj_pop_dist.csv", DataFrame);
states_idx = [2, 3, 9, 14, 15, 22, 25, 26, 35, 46]
A_pop = Matrix(A_pop[:, 2:end]);
minA = minimum([A_pop[i, j] for i in axes(A_pop, 1), j in axes(A_pop, 2) if i != j])
maxA = maximum(A_pop)
A_norm = (A_pop .- minA) ./ (maxA - minA)
[A_norm[i, i] = 0.0 for i in axes(A_norm, 1)]
A = A_norm[states_idx, states_idx]

# Build Graph
g = build_GNNGraph(A)

# Check Data Size
X_full = g.ndata.x
println("Full Dataset Size: ", size(X_full))
println("Variables: ", size(X_full, 1))
println("Time Steps: ", size(X_full, 2))
println("Nodes: ", size(X_full, 3))
