# Sanity Check Script for Test 8
# Purpose: Verify that the model's success is due to Training and Topology.

using JLD2, ComponentArrays, Lux
using CSV, DataFrames, Statistics, Plots, Measures
using GNNLux, Graphs, NPZ
using DifferentialEquations, SciMLSensitivity, Random, LinearAlgebra, CubicSplines

# Setup (Copied from evaluate_unseen.jl)
trained_states = [
    "FL", "IL", "NC", "CA", "NJ", "GA", "OH", "TX", "NY", "VA",
    "WA", "MI", "MA", "AZ", "CO", "MD", "WI", "MN", "SC", "KY",
    "CT", "MO", "IN", "NM", "NV"
]
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states = names(df_adj)[2:end]
unseen_states = setdiff(all_states, trained_states)
features_raw = NPZ.npzread("Data/data_filtered.npz")

# Helper to load data for a specific list of states
function load_data_and_graph(state_list, shuffle_graph=false)
    # Data
    valid_states = []
    X_tensor = []
    for s in state_list
        if haskey(features_raw, s)
            push!(valid_states, s)
            push!(X_tensor, features_raw[s])
        end
    end
    n_nodes = length(valid_states)
    X_tensor = cat(X_tensor..., dims=3) # (Vars, Time, Nodes)
    X_norm = log.(X_tensor .+ 1.0)

    # Graph
    csv_cols = names(df_adj)[2:end]
    real_idxs = [findfirst(==(s), csv_cols) for s in valid_states]

    # SHUFFLE ATTACK: Permute the indices used to fetch ADJACENCY, but keep Feature order same.
    # This means Node i has Features of State i, but Neighbors of State perm[i].
    if shuffle_graph
        println("!!! SHUFFLING TOPOLOGY !!!")
        rng_shuf = Random.MersenneTwister(1234)
        real_idxs = shuffle(rng_shuf, real_idxs)
    end

    adj = Matrix(df_adj[:, 2:end])[real_idxs, real_idxs]
    adj = (adj ./ maximum(adj))
    for i in 1:n_nodes
        adj[i, i] = 0.0
    end
    g = GNNGraph(adj + I)

    tsteps = Float32.(collect(0:400))
    splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:4, n in 1:n_nodes]

    return valid_states, X_norm, g, splines
end

# Model Def (Standard)
latent_dim = 3
nin_tot = 1 + 3 + latent_dim
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end
Lux.initialparameters(r::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(r::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=r)
(d::FrozenDropout)(x, ps, st) = st.mask === nothing ? (x, st) : (x .* st.mask, st)
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
        GNNLux.GraphConv(nin => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nhidden, tanh; aggr=mean), FrozenDropout(drop_p),
        GNNLux.GraphConv(nhidden => nout; aggr=mean)
    )
end
# Basic methods
Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialparameters(rng, m.layer1), drop1=Lux.initialparameters(rng, m.drop1),
    layer2=Lux.initialparameters(rng, m.layer2), drop2=Lux.initialparameters(rng, m.drop2),
    layer3=Lux.initialparameters(rng, m.layer3), drop3=Lux.initialparameters(rng, m.drop3),
    layer4=Lux.initialparameters(rng, m.layer4)
)
Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = (
    layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1),
    layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2),
    layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3),
    layer4=Lux.initialstates(rng, m.layer4)
)
function (m::ExplicitGNN)(g, x, ps, st)
    x, st_l1 = m.layer1(g, x, ps.layer1, st.layer1)
    x, st_d1 = m.drop1(x, ps.drop1, st.drop1)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)
    x = typeof(x) <: Tuple ? x[1] : x
    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
end

function predict(model, ps, st, g, u0, tsteps, splines, n_nodes)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        col_t = map(s -> s(t), splines)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, col_t, latents)
        y, _ = model(g, model_input, p_ode.gnn, st)
        return vec(y)
    end
    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps, saveat=tsteps, reltol=1e-4, abstol=1e-4) # Fast tolerances
    return permutedims(reshape(reduce(hcat, sol.u), 1, n_nodes, length(tsteps)), (1, 3, 2))
end

# === RUN EXPERIMENTS ===
rng = Random.default_rng()
gnn_model = ExplicitGNN(nin_tot, 64, 1, 0.0)
_, st = Lux.setup(rng, gnn_model)

# Load real parameters
ckpt = "Resultados/test-8/checkpoints/params_test8_25s.jld2"
@load ckpt ps_trained

# Define subset for speed (Focus on interesting ones)
test_nodes = ["PA", "TN", "OR", "LA", "ME", "WY"]

println("--- Experiment 1: Baseline (Correct Graph, Trained Weights) ---")
nodes_1, X_1, g_1, spl_1 = load_data_and_graph(test_nodes, false)
ps_1 = ComponentArray(gnn=ps_trained.gnn, latent_features=randn(rng, Float32, 3, length(test_nodes)) * 0.1)
pred_1 = predict(gnn_model, ps_1, st, g_1, X_1[1, 1, :], 0:400, spl_1, length(test_nodes))

println("--- Experiment 2: Untrained GNN (Correct Graph, Random Weights) ---")
ps_rand_gnn = Lux.initialparameters(rng, gnn_model)
ps_2 = ComponentArray(gnn=ps_rand_gnn, latent_features=randn(rng, Float32, 3, length(test_nodes)) * 0.1)
pred_2 = predict(gnn_model, ps_2, st, g_1, X_1[1, 1, :], 0:400, spl_1, length(test_nodes))

println("--- Experiment 3: Shuffled Topology (Wrong Graph, Trained Weights) ---")
nodes_3, X_3, g_3, spl_3 = load_data_and_graph(test_nodes, true) # SHUFFLE
# Use same Features and Latents, just different G
pred_3 = predict(gnn_model, ps_1, st, g_3, X_3[1, 1, :], 0:400, spl_3, length(test_nodes))

println("--- Experiment 4: Full US Graph (49 States) ---")
# Load ALL states
valid_all, X_all, g_all, spl_all = load_data_and_graph(all_states, false)
n_all = length(valid_all)
ps_4 = ComponentArray(gnn=ps_trained.gnn, latent_features=randn(rng, Float32, 3, n_all) * 0.1) # New latents for everyone? 
# Ideally we reuse trained latents for trained nodes?
# Too complex for quick sanity check?
# Let's just use random latents for ALL to be fair (Zero-Shot all).
# Or better: random latents for all 49 states.
pred_4 = predict(gnn_model, ps_4, st, g_all, X_all[1, 1, :], 0:400, spl_all, n_all)

# Evaluate
function get_mse_general(pred, X, state_list, target_state)
    idx = findfirst(==(target_state), state_list)
    if isnothing(idx)
        return NaN
    end
    test_idx = 181:401
    return mean(abs2, pred[1, test_idx, idx] .- X[1, test_idx, idx])
end

println("\n=== SANITY CHECK RESULTS (Test MSE) ===")
println("State | Subgraph(Base) | Untrained | Shuffled | Full_Graph | Verdict")
println("------+----------------+-----------+----------+------------+--------")

results = []
for state in test_nodes
    mse_base = get_mse_general(pred_1, X_1, nodes_1, state)
    mse_rand = get_mse_general(pred_2, X_1, nodes_1, state)
    mse_shuf = get_mse_general(pred_3, X_3, nodes_1, state)
    mse_full = get_mse_general(pred_4, X_all, valid_all, state)

    gap_rand = mse_rand / mse_base

    # Verdict logic
    verdict = (gap_rand > 2.0) ? "PASS" : "FAIL"

    println("$state    | $(round(mse_base, digits=3))          | $(round(mse_rand, digits=3))     | $(round(mse_shuf, digits=3))    | $(round(mse_full, digits=3))      | $verdict")
    push!(results, (State=state, Base=mse_base, Rand=mse_rand, Shuf=mse_shuf, Full=mse_full))
end

CSV.write("Resultados/test-8/sanity_results_v2.csv", DataFrame(results))
