# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using JSON

rng = Random.default_rng()
Random.seed!(rng, 42)

# 1. SETUP
# ==============================================================================
# 1. SETUP
# ==============================================================================
# include("../Train/preprocessing.jl") <--- REMOVED (Hardcoded filters)

using NPZ

# Data Loading Logic (Reconstructed to allow ANY state)
println("Loading raw datasets...")
features_raw = npzread("Data/data_filtered.npz")
# features_raw is a Dict{String, Matrix} where Key="OH", Value=Matrix (Vars x Time)

# Adjacency
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states_adj = String.(df_adj[:, 1])
A_pop_raw = Matrix(df_adj[:, 2:end])

# Normalize Adjacency (Same logic as preprocessing.jl)
minA = minimum([A_pop_raw[i, j] for i in axes(A_pop_raw, 1) for j in axes(A_pop_raw, 2) if i != j])
maxA = maximum(A_pop_raw)
A_norm = (A_pop_raw .- minA) ./ (maxA - minA)
[A_norm[i, i] = 0.0 for i in axes(A_norm, 1)]
A_hat_full = A_norm + I # Add self-loops

# INDICES FOR ZERO-SHOT TEST
# Test States: PA, MI, WA, MA, AZ
test_states_names = ["PA", "MI", "WA", "MA", "AZ"]
println("Testing Zero-Shot on: ", test_states_names)

# Get indices in Adjacency Matrix
test_indices = [findfirst(==(s), all_states_adj) for s in test_states_names]
println("Test Indices: ", test_indices)

# Filter Data (X)
# X shape in preprocessing.jl was (Vars, Time, Nodes). 
# features_raw[state] gives (Vars, Time) ?? Let's check dimension order.
# In preprocessing, it was: X = Float64.(g.ndata.x)
# In build_GNNGraph: attributes = stack(ordered_matrices)
# If stack aligns along last dim, then it is (Dims, Nodes).
# Wait, let's assume features_raw[state] is (Vars, Time).
# Let's inspect one.
# Filter Data (X)
# Data is (Vars, Time) e.g. (4, 401)
sample_feat = features_raw[test_states_names[1]]
n_vars, n_time_full = size(sample_feat)

# User Request: Full Coverage (Train + Test Days)
n_time = n_time_full
println("Prediction Horizon: $n_time days")

num_test = length(test_states_names)

X_test = zeros(n_vars, n_time, num_test)
for (i, state) in enumerate(test_states_names)
    # Take full timeline
    X_test[:, :, i] = features_raw[state][:, 1:n_time]
end

# Log Normalize
X_test = log.(X_test .+ 1)

u0_test = X_test[1, 1, :]
println("u0 (Initial Conditions): ", u0_test)
println("Test Data Size: ", size(X_test))

# Filter Adjacency
A_temp = A_hat_full[test_indices, test_indices]

# SPECTRAL NORMALIZATION (Crucial for GNN stability)
# The training graph likely had max eigenvalue close to 1.
# Slicing can change this. We force it back to 1.
eig_max = maximum(abs.(eigvals(A_temp)))
println("Max Eigenvalue before norm: ", eig_max)
A_test = A_temp ./ eig_max
println("Adjacency normalized by spectral radius.")

g_test = GNNGraph(A_test)
println("Test Graph Nodes: ", g_test.num_nodes)

# Create Splines for Test Group (Covariates are present!)
tsteps = collect(0.0:1.0:(size(X_test, 2)-1))
# === MODEL DEFINITION (Must match Test 3 / ExplicitGNN) ===
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
    if st.mask === nothing
        return x, st
    else
        return x .* st.mask, st
    end
end

struct ExplicitGNN{L1,D1,L2,D2,L3,D3,L4} <: Lux.AbstractLuxLayer
    layer1::L1
    drop1::D1
    layer2::L2
    drop2::D2
    layer3::L3
    drop3::D3
    layer4::L4
end

function ExplicitGNN(nin, nhidden, nout, drop_p)
    return ExplicitGNN(
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean),
        FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean),
        FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean),
        FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN)
    return (
        layer1=Lux.initialparameters(rng, m.layer1),
        drop1=Lux.initialparameters(rng, m.drop1),
        layer2=Lux.initialparameters(rng, m.layer2),
        drop2=Lux.initialparameters(rng, m.drop2),
        layer3=Lux.initialparameters(rng, m.layer3),
        drop3=Lux.initialparameters(rng, m.drop3),
        layer4=Lux.initialparameters(rng, m.layer4)
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN)
    return (
        layer1=Lux.initialstates(rng, m.layer1),
        drop1=Lux.initialstates(rng, m.drop1),
        layer2=Lux.initialstates(rng, m.layer2),
        drop2=Lux.initialstates(rng, m.drop2),
        layer3=Lux.initialstates(rng, m.layer3),
        drop3=Lux.initialstates(rng, m.drop3),
        layer4=Lux.initialstates(rng, m.layer4)
    )
end

function (m::ExplicitGNN)(g, x, ps, st)
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)
    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)
    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    st_new = (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
    return x, st_new
end

test_splines = [CubicSpline(tsteps, @view X_test[v, :, n]) for v in 2:size(X_test, 1), n in 1:size(X_test, 3)]

# 2. LOAD MODEL
# ==============================================================================
println("Loading parameters from Params/par_opt_test3.jld2...")
@load "Params/par_opt_test3.jld2" ps_trained

# DETECT LATENT DIMENSION
latent_dim_detected = size(ps_trained.latent_features, 1)
println("Detected Latent Dimension: ", latent_dim_detected)

latent_dim = latent_dim_detected
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1

# Instantiate with Test 3 Config (Width 32, Dropout 0.05)
gnn = ExplicitGNN(nin_tot, 32, nout, 0.05)

# 3. ADAPTING PARAMETERS
# ==============================================================================
# TRANSFER LEARNING STRATEGY
train_latents = ps_trained.latent_features
μ_lat = mean(train_latents)
σ_lat = std(train_latents)

println("Initializing new latents with Train Stats: Mean=$μ_lat, Std=$σ_lat")
# Random initialization matching training distribution
new_latents = (randn(rng, latent_dim, length(test_indices)) .* σ_lat) .+ μ_lat

ps_test = ComponentArray(gnn=ps_trained.gnn, latent_features=new_latents) |> f64

# 4. PREDICTION FUNCTION
# ==============================================================================
function predict(model, p, st, u0, tsteps, splines, graph)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, graph.num_nodes)
        cov_matrix = map(s -> s(t), splines)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)
        y, _ = model(graph, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps, reltol=1e-5, abstol=1e-6)

    if length(sol.t) != length(tsteps)
        println("Integration failed.")
        return nothing
    end

    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, graph.num_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

# 5. RUN & PLOT
# ==============================================================================
println("Running Zero-Shot Prediction...")
_, st_test = Lux.setup(rng, gnn) # Fresh state
pred = predict(gnn, ps_test, st_test, u0_test, tsteps, test_splines, g_test)

# De-normalize
pred_denorm = exp.(pred) .- 1
X_test_denorm = exp.(X_test) .- 1

mkpath("Resultados/test-5/plots")

for (i, state_name) in enumerate(test_states_names)
    p = plot(title="Zero-Shot Generalization: $state_name", xlabel="Days", ylabel="New Cases")

    # Real
    real = X_test_denorm[1, :, i]
    scatter!(p, tsteps, real, label="Real Data", color=:blue, ms=2, alpha=0.5)

    # Pred
    pr = pred_denorm[1, :, i]
    plot!(p, tsteps, pr, label="Prediction", color=:red, lw=2)

    # Mark Training Horizon (Day 180)
    vline!(p, [180], label="Training Horizon", color=:black, linestyle=:dash)

    savefig(p, "Resultados/test-5/plots/$(state_name)_zero_shot.png")
end

println("Plots saved to Resultados/test-5/plots/")
