# ==============================================================================
# 1. LOAD DEPENDENCIES
# ==============================================================================
println("Loading dependencies...")
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, NPZ
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines

# ==============================================================================
# 2. BOOTSTRAP DATA & ADJACENCY
# ==============================================================================
println("\nBootstrapping Data & Adjacency...")

train_states = [
    "AL", "CA", "CO", "CT", "DC", "DE", "FL", "IA", "ID", "IL", "IN", "KS", "KY",
    "LA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "NE", "NH", "NJ", "NM",
    "NY", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WI", "WV", "WY"
]

n_nodes = length(train_states)
features_raw = NPZ.npzread("../Data/data_all.npz")
n_vars, n_times = size(features_raw["NY"])
X_tensor_all = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(train_states)
    X_tensor_all[:, :, i] = features_raw[state]
end

# Parse covariates
idx_list = [1,2,3,6] 
X_tensor = X_tensor_all[idx_list, :, :]

# Parse lags
lags = [7,7,14]
if length(lags) != length(idx_list)-1
    error("ERROR: You provided $(length(lags)) lags for $(length(idx_list)-1) covariates!")
end
max_lag = isempty(lags) ? 0 : maximum(lags)
new_T = n_times - max_lag
X_aligned = zeros(Float32, length(idx_list), new_T, n_nodes)
X_aligned[1, :, :] = X_tensor[1, (max_lag + 1):n_times, :]

for i in eachindex(lags)
    l = lags[i]
    var_idx = i + 1 
    start_idx = max_lag - l + 1
    end_idx = n_times - l
    X_aligned[var_idx, :, :] = X_tensor[var_idx, start_idx:end_idx, :]
end

# Normalize and Spline
X_norm = log.(X_aligned .+ 1.0f0)
tsteps = Float32.(collect(0:new_T-1))
spls = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

# Adjacency
df_adj_full = CSV.read("../Data/flow_mat_exp_gravity_fitted.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
name_to_index = Dict(name => i for (i, name) in enumerate(col_names))
indices = [name_to_index[s] for s in train_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]

# ADJACENCY MATRIX FOR GCNConv LAYER
A = copy(adj_sub)
A[diagind(A)] .= 0.0
src_idx, dst_idx, vals = findnz(sparse(Float32.(A)))
g = GNNGraph(src_idx, dst_idx, vals)

# ADJACENCY MATRIX FOR CUSTOM LAYER
# function col_normalize(mat; self_loops=true)
#     # matrix has flows origin state on rows, destination on columns
#     # normalize arriving signals on columns to one
#     mat = copy(mat)
#     mat[diagind(mat)] .= self_loops ? 1.0 : 0.0  
#     col_sums = sum(mat, dims=1)
#     return Float32.(mat ./ (col_sums .+ 1e-8))
# end

# A = col_normalize(adj_sub; self_loops=true)

# ==============================================================================
# 3. EXECUTE RUN
# ==============================================================================
println("\nLoading training function...")
include("training_custom.jl")

println("\nStarting training run...")

save_path = length(ARGS) > 0 ? ARGS[1] : "Output"
println("Outputs will be saved to: $save_path")
mkpath(save_path) 

run_start = time()
seed = rand(Int)

# Training function call 

# GCNConv layer
train_model(X_norm, g, tsteps, spls, save_path; seed=seed)
# Custom layer
#train_model(X_norm, A, tsteps, spls, save_path; seed=seed)

run_elapsed = time() - run_start
println("\nTotal wall time: $(round(run_elapsed, digits=2)) seconds")