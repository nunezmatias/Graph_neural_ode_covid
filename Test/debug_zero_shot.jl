using JLD2, ComponentArrays, Statistics, LinearAlgebra, CSV, DataFrames, Lux

# 1. Load Model Params
@load "Params/par_opt_new.jld2" ps_trained
latents = ps_trained.latent_features

println("=== Latent Variable Stats (Training) ===")
println("Mean: ", mean(latents))
println("Std:  ", std(latents))
println("Min:  ", minimum(latents))
println("Max:  ", maximum(latents))
println("Shape: ", size(latents))

# 2. Check Adjacency Slicing Logic
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
A_pop_raw = Matrix(df_adj[:, 2:end])

# Standard Normalization (as in preprocessing)
minA = minimum([A_pop_raw[i, j] for i in axes(A_pop_raw, 1) for j in axes(A_pop_raw, 2) if i != j])
maxA = maximum(A_pop_raw)
A_norm = (A_pop_raw .- minA) ./ (maxA - minA)
[A_norm[i, i] = 0.0 for i in axes(A_norm, 1)]
A_full = A_norm + I

# Test Indices (PA, MI, WA, MA, AZ)
test_states = ["PA", "MI", "WA", "MA", "AZ"]
all_states = String.(df_adj[:, 1])
idxs = [findfirst(==(s), all_states) for s in test_states]

# Slice
A_slice = A_full[idxs, idxs]

println("\n=== Adjacency Matrix Stats (Test Subgraph) ===")
println("Row Sums (Should be closest to 1 for stability):")
println(sum(A_slice, dims=2))

println("\nEigenvalues of Sliced Matrix (Max should be ~1):")
max_eig = maximum(abs.(eigvals(A_slice)))
println("Max Eigenvalue: ", max_eig)

if max_eig < 0.5
    println(">>> CRITICAL: Graph is disconnected/weak. Signal will vanish.")
end
