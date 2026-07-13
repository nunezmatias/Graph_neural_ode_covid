# run_ensemble.jl (Master Script)
using Distributed
using ParallelDataTransfer

# 1. SETUP WORKERS
rmprocs(workers())
println("Start worker setup...")
num_sim = 10
threads_per_worker = 4
if nprocs() == 1
    addprocs(num_sim; exeflags=["--threads=$(threads_per_worker)", "--project=@."])
end

# 2. LOAD DEPENDENCIES (Everywhere)
println("Loading dependencies on all workers...")
@everywhere begin
    using ParallelDataTransfer
    using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, NPZ
    using Graphs, Lux, GNNLux
    using DifferentialEquations, DiffEqFlux
    using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
    using LinearAlgebra, Statistics, Random, Plots, CubicSplines
    BLAS.set_num_threads(Threads.nthreads())
end

# 3. LOAD DATA ON MASTER ONLY 
println("\n[Master] Bootstrapping Data & Adjacency...")

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
SHARED_DATA = log.(X_aligned .+ 1.0f0)
SHARED_TSTEPS = Float32.(collect(0:new_T-1))
SHARED_SPLINES = [CubicSpline(SHARED_TSTEPS, @view SHARED_DATA[v, :, n]) for v in 2:size(SHARED_DATA, 1), n in 1:n_nodes]

# Adjacency
df_adj_full = CSV.read("../Data/fitted_matrix/flow_mat_exp_gravity_fitted.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
name_to_index = Dict(name => i for (i, name) in enumerate(col_names))
indices = [name_to_index[s] for s in train_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]

function controlled_normalize(A; alpha=0.6f0)
    A_no_diag = copy(A)
    # Remove self-loops entirely for the neighbor calculation
    A_no_diag[diagind(A_no_diag)] .= 0.0

    # Normalize ONLY the neighbor weights
    col_sums = sum(A_no_diag, dims=1)
    A_neigh_norm = Float32.(A_no_diag ./ (col_sums .+ 1e-8))

    # Create an Identity matrix
    I_mat = Matrix{Float32}(I, size(A)...)

    # Blend: alpha * Self + (1 - alpha) * Neighbors
    return alpha .* I_mat .+ (1.0f0 - alpha) .* A_neigh_norm
end

SHARED_A = controlled_normalize(adj_sub; alpha=0.5f0)

println("Symmetric? $(issymmetric(SHARED_A))")         
println("Self-loops? $(all(diag(SHARED_A) .> 0))")       

# 4. BROADCAST VARIABLES (Master -> Workers)
println("\n[Master] Broadcasting data to workers...")
passobj(1, workers(), [:SHARED_DATA, :SHARED_A, :SHARED_TSTEPS, :SHARED_SPLINES])

# 5. DEFINE TRAINING FUNCTION (Everywhere)
println("[Master] Loading training function on workers...")
@everywhere include("function_train_custom_v2.jl")

# 6. RUN
println("\n[Master] Starting ensemble training...")

ensemble_start = time()

save_path = "Results_custom_V2"
mkdir(save_path)

results = pmap(1:num_sim) do i
    seed = rand(Int)
    println("Master dispatching run $i to Worker $(myid())")
    train_model(seed, i, save_path)
end

ensemble_elapsed = time() - ensemble_start
println("\nTotal wall time: $(round(ensemble_elapsed, digits=2)) seconds")
