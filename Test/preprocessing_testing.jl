using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays, JSON;
using Graphs, Lux, GNNLux;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

# Load my modules;
push!(LOAD_PATH, "./Utils/GraphCreator/");
using GNNGraph_mod;

#################################################
################## SETUP ########################
#################################################

# import adjacency matrix
A_pop = CSV.read("./Data/adj_pop_dist.csv", DataFrame);
states = String.(A_pop[:, "Column1"]);
# take states subset
states_idx = [2, 3, 9, 14, 15, 22, 25, 26, 35, 46]
states = states[states_idx]
A_pop = Matrix(A_pop[:, 2:end]);

# normalize pop adjacency
minA_pop = minimum([A_pop[i, j] for i in axes(A_pop, 1) for j in axes(A_pop, 2) if i != j])
maxA_pop = maximum(A_pop)
A_pop_normalized = (A_pop .- minA_pop) ./ (maxA_pop - minA_pop)
[A_pop_normalized[i, i] = 0.0 for i in axes(A_pop_normalized, 1)]

# USA populations
df_pop = CSV.read("./Data/us_pop_by_state.csv", DataFrame)
ordered_df = df_pop[[findfirst(==(state), df_pop.state_code) for state in states], :]
census = ordered_df[!, "2020_census"]

function norm_adj(A)
    A = A + I # add self-loops
    # No spectral normalization here, let GNNConv(aggr=mean) handle it
    return A
end

A_hat = norm_adj(A_pop_normalized)

# Take smaller subset of adjacency
A = A_hat[states_idx, states_idx]

# matrix checks
sym_diff = A_hat - A_hat'                     # elementwise difference
max_abs_diff = maximum(abs.(sym_diff))  # manual symmetry check 
maximum(abs.(sum(A_hat, dims=2)))  # rows should have similar magnitudes (about 1)

eigvals_A = eigvals(A_hat)           # Compute eigenvalues
eig_abs = abs.(eigvals_A)
λ_max = maximum(eig_abs)             # Maximum absolute eigenvalue
println("Eigenvalues condition satisfied? ", λ_max ≤ 1.05)

# make GNN graph;
g = build_GNNGraph(A)

# split dataset
n_train = 180
n_test = 221
n_total = n_train + n_test

X = Float64.(g.ndata.x) # var x time x nodes
X_train = X[:, 1:n_train, :]

# Log normalization (Match training)
X_total = X[:, 1:n_total, :]
X_total_norm = log.(X_total .+ 1)

# initial condition
u0 = X_total_norm[1, 1, :]

# time
tsteps_test = collect(0.0:1.0:n_total-1)
tspan = (tsteps_test[1], tsteps_test[end])

# dates for plotting
dates = JSON.parsefile("./Data/times.json")

# data size
F, T, N = size(X_total)

# Define matrix of splines functions: Num_var x Num_nodes
# variables: Cli_cmnty, doctor_visits, events_indoor

splines_test = [CubicSpline(tsteps_test, @view X_total_norm[v, :, n]) for v in 2:size(X_total_norm, 1), n in 1:size(X_total_norm, 3)]
covariate_splines = splines_test # Also define this name for fallback

