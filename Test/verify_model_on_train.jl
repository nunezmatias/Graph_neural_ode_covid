# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using JSON
using NPZ

rng = Random.default_rng()
Random.seed!(rng, 42)

# 1. SETUP DATA (Similar to predict_zero_shot)
println("Loading raw datasets...")
features_raw = npzread("Data/data_filtered.npz")
df_adj = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states = String.(df_adj[:, 1])
A_pop_raw = Matrix(df_adj[:, 2:end])

# Normalize Adjacency
minA = minimum([A_pop_raw[i, j] for i in axes(A_pop_raw, 1) for j in axes(A_pop_raw, 2) if i != j])
maxA = maximum(A_pop_raw)
A_norm = (A_pop_raw .- minA) ./ (maxA - minA)
[A_norm[i, i] = 0.0 for i in axes(A_norm, 1)]
A_full = A_norm + I

# TEST STATE: CA (California) - KNOWN TO BE IN TRAIN SET
target_state = "CA"
idx = findfirst(==(target_state), all_states)
println("Verifying Model on Training State: $target_state (Index $idx)")

# Data
feat = features_raw[target_state] # (Vars, Time)
n_vars, n_time_full = size(feat)
n_time = 180
X_target = zeros(n_vars, n_time, 1)
X_target[:, :, 1] = feat[:, 1:n_time]
X_target = log.(X_target .+ 1)

u0 = X_target[1, 1, :]

# Graph (Single Node Self-Loop)
# Training was on a graph of 10 nodes. 
# Ideally we should reconstruct the FULL 10-node Training Graph to reproduce training conditions exactly.
# Train States: FL, IL, NC, CA, NJ, GA, OH, TX, NY, VA
train_states = ["FL", "IL", "NC", "CA", "NJ", "GA", "OH", "TX", "NY", "VA"]
train_idxs = [findfirst(==(s), all_states) for s in train_states]

println("Reconstructing Test 2 Training Graph (10 nodes)...")
A_train = A_full[train_idxs, train_idxs]
g_train = GNNGraph(A_train)

# Load Model
println("Loading parameters from Params/par_opt_test3.jld2...")
@load "Params/par_opt_test3.jld2" ps_trained
latents = ps_trained.latent_features # shape (3, 10)

# Verify Latent Size matches
if size(latents, 2) != 10
    println("ERROR: Saved latents dimension $(size(latents)) does not match expected 10 nodes.")
else
    println("Latent dimensions match (10 nodes).")
end

# Prepare Input Data (All 10 states)
X_train_all = zeros(n_vars, n_time, 10)
for (i, s) in enumerate(train_states)
    X_train_all[:, :, i] = features_raw[s][:, 1:n_time]
end
X_train_all = log.(X_train_all .+ 1)

tsteps = collect(0.0:1.0:179.0)
splines = [CubicSpline(tsteps, @view X_train_all[v, :, n]) for v in 2:4, n in 1:10]

# Model Config
latent_dim = size(latents, 1)
nin_tot = 1 + 3 + latent_dim
nout = 1
# === MODEL DEFINITION FROM Train/model_opt.jl ===
# Frozen Dropout Layer
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

# Model Config
latent_dim = size(latents, 1) # Detect from loaded file
nin_tot = 1 + 3 + latent_dim
nout = 1

# Instantiate with same dropout (p=0.05 as in model_opt.jl)
gnn = ExplicitGNN(nin_tot, 32, nout, 0.05)

# Prediction Function
function predict(model, p, st, u0, tsteps, splines, graph)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, graph.num_nodes)
        cov_matrix = map(s -> s(t), splines)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)
        y, _ = model(graph, model_input, p_ode.gnn, st)
        return vec(y)
    end
    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    solve(prob, Tsit5(), p=p, saveat=tsteps, reltol=1e-5, abstol=1e-6)
end

println("Running prediction on Training Set...")
_, st = Lux.setup(rng, gnn)
u0_train = X_train_all[1, 1, :]
sol = predict(gnn, ps_trained, st, u0_train, tsteps, splines, g_train)

# Check CA prediction (Index 4 in train_states)
ca_idx = 4
pred_matrix = reduce(hcat, sol.u) # (Nodes, Time) - wait, dudt returns vec(y) which is Nodes*1
# sol.u is vector of vectors.
# Each element is length 10.
pred_ca = [u[ca_idx] for u in sol.u]
real_ca = X_train_all[1, :, ca_idx]

# Compare
mse = mean(abs2, pred_ca .- real_ca)
println("MSE on CA (Train State): ", mse)

# Plot
p = plot(title="Sanity Check: CA (Train State)", legend=:topleft)
plot!(p, tsteps, real_ca, label="Real (Log)", lw=2)
plot!(p, tsteps, pred_ca, label="Pred (Log)", lw=2, linestyle=:dash)
savefig(p, "Resultados/test-5/plots/sanity_check_CA.png")
println("Sanity check plot saved.")
