# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using NPZ

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 42)

println("Running Test 8 (25 States) with ", Threads.nthreads(), " threads")

# -------------------------------------------------------------
# 1. Configuration & Hyperparameters (Matches Test 2)
# -------------------------------------------------------------

# Refined Curriculum from Test 2
curriculum_steps = [5, 10, 20, 40, 60, 90, 120, 150, 180]
epochs_per_stage = [50, 50, 100, 150, 200, 250, 300, 400, 1500]

# Learning Rate Schedule (Matches Test 2)
lr_schedule = [
    Dict(1 => 5e-4), # Stage 1
    Dict(1 => 5e-4), # Stage 2
    Dict(1 => 5e-4), # Stage 3
    Dict(1 => 5e-4), # Stage 4
    Dict(1 => 5e-4), # Stage 5
    Dict(1 => 5e-4), # Stage 6
    Dict(1 => 5e-4), # Stage 7
    Dict(1 => 5e-4), # Stage 8
    Dict(1 => 5e-4, 500 => 1e-5, 1000 => 1e-6) # Stage 9 (Deep fine-tuning)
]

opt = Optimisers.AdamW(eta=5e-4, lambda=1.0f-4)

# -------------------------------------------------------------
# 2. Data Loading (Specific 25 States)
# -------------------------------------------------------------
println("Loading specific 25 states for Test 8...")

target_states = [
    "FL", "IL", "NC", "CA", "NJ", "GA", "OH", "TX", "NY", "VA", # Original (10)
    "WA", "MI", "MA", "AZ", "CO", "MD", "WI", "MN", "SC", "KY", # Zero-Shot High Perf (10)
    "CT", "MO", "IN", "NM", "NV"                                # New Stable Replacements (5)
]

# 2.1 Load Features (using NPZ archive)
features_raw = NPZ.npzread("../../Data/data_filtered.npz")

# Verify availability
missing_states = [s for s in target_states if !haskey(features_raw, s)]
if !isempty(missing_states)
    error("Missing states in NPZ: $missing_states")
end

# Stack features: shape (Nodes) of (Vars x Time) -> (Vars x Time x Nodes)
# We assume raw features are (4, Time) or (Time, 4). Let's check dimensions.
example_feat = features_raw["NY"]
println("Raw feature shape (NY): ", size(example_feat))
# Assuming shape is (4, Time). If (Time, 4), we permute.
# Standard in this repo seems to be (Vars, Time)

# We construct the 3D tensor
n_vars, n_times = size(example_feat)
# Check if (Time, Vars)
if n_vars > n_times
    # Likely (Time, Vars)?? No, usually Time is large (400+). Vars is ~4.
    # If 4 > 400, that's weird. 
    # Let's assume (Vars, Time) as per `data_clean.jl` implies GNN structure.
    nothing
end

n_nodes = length(target_states)
X_tensor = zeros(Float32, n_vars, n_times, n_nodes)

for (i, state) in enumerate(target_states)
    X_tensor[:, :, i] = features_raw[state]
end

# 2.2 Normalization (Log Cases)
# Feature 0 is usually cases. Others are covariates.
# We normalize ONLY cases with Log(x+1). Covariates usually pre-normalized?
# Let's follow `preprocessing.jl`: X_norm = log.(X_true .+ 1)
# Checking preprocessing.jl line 64: `X_norm = log.(X .+ 1)` applies to ALL features?
# Line 79: `covariate_splines = ... @view X_norm[v, :, n]`
# This implies ALL features are log-transformed. 
X_norm = log.(X_tensor .+ 1.0) # Apply global log transform

# 2.3 Build Adjacency for 25 States
df_adj_full = CSV.read("../../Data/adj_pop_dist.csv", DataFrame)
# Extract submatrix
col_names = names(df_adj_full)[2:end] # Skip first col (labels)
indices = [findfirst(x -> x == s, col_names) for s in target_states]

adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]

# Normalize Adjacency
minA = minimum([adj_sub[i, j] for i in 1:n_nodes, j in 1:n_nodes if i != j])
maxA = maximum(adj_sub)
adj_norm = (adj_sub .- minA) ./ (maxA - minA)
for i in 1:n_nodes
    adj_norm[i, i] = 0.0
end
adj_final = adj_norm + I # Add self loops

# Build Graph
g = GNNGraph(adj_final)

# 2.4 Prepare Splines
tsteps = Float32.(collect(0:n_times-1))
# Features: 1=Cases, 2=Cov1, 3=Cov2, 4=Cov3. 
# We target variable 1. Input to model includes 1 (Auto-regressive) + 3 (Covs) + Latent.
# Preprocessing.jl defines splines for v in 2:end.
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

u0 = X_norm[1, 1, :] # Initial condition (Cases day 0)

println("Data loaded. Nodes: $n_nodes, Time: $n_times, Features: $n_vars")
println("Splines ready.")

# -------------------------------------------------------------
# 3. Model Definition (Width 64)
# -------------------------------------------------------------

latent_dim = 3
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1

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

# MODIFIED: Hidden Dim = 64
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
    x, st_d1 = (x, st_d1) # fix tuple unpacking if dropout returns tuple

    x, st_l2 = m.layer2(g, x, ps.layer2, st.layer2)
    x, st_d2 = m.drop2(x, ps.drop2, st.drop2)

    x, st_l3 = m.layer3(g, x, ps.layer3, st.layer3)
    x, st_d3 = m.drop3(x, ps.drop3, st.drop3)

    x, st_l4 = m.layer4(g, x, ps.layer4, st.layer4)
    return x, (layer1=st_l1, drop1=st_d1, layer2=st_l2, drop2=st_d2, layer3=st_l3, drop3=st_d3, layer4=st_l4)
end

function sample_dropout_masks(model, st, x_shape)
    d1, d2, d3 = model.drop1, model.drop2, model.drop3
    # Use 64 as hidden dim
    mask1 = rand(st.drop1.rng, Float32, (64, x_shape[2])) .> d1.p
    mask2 = rand(st.drop2.rng, Float32, (64, x_shape[2])) .> d2.p
    mask3 = rand(st.drop3.rng, Float32, (64, x_shape[2])) .> d3.p

    st_d1 = (mask=mask1 ./ (1 - d1.p), rng=st.drop1.rng)
    st_d2 = (mask=mask2 ./ (1 - d2.p), rng=st.drop2.rng)
    st_d3 = (mask=mask3 ./ (1 - d3.p), rng=st.drop3.rng)

    return (
        layer1=st.layer1, drop1=st_d1,
        layer2=st.layer2, drop2=st_d2,
        layer3=st.layer3, drop3=st_d3,
        layer4=st.layer4
    )
end

# Initialize Model (Width=64)
println("Initializing GNN (Width 64)...")
gnn = ExplicitGNN(nin_tot, 64, nout, 0.0)
ps_gnn, st_gnn = Lux.setup(rng, gnn)

latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes) |> f64
ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features) |> f64

# -------------------------------------------------------------
# 4. Training (Curriculum with Early Stopping)
# -------------------------------------------------------------

function loss_curriculum(model, p, st, data, tsteps, splines)
    u0_data = data[1, 1, :]
    tspan = (tsteps[1], tsteps[end])
    n_nodes = size(data, 3)
    st_frozen = sample_dropout_masks(model, st, (64, n_nodes)) # Frozen Dropout

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, n_nodes)
        cov_matrix = map(s -> s(t), splines)
        latents = p_ode.latent_features
        model_input = vcat(u_reshaped, cov_matrix, latents)
        y, _ = model(g, model_input, p_ode.gnn, st_frozen)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0_data), tspan)
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps,
        sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()),
        reltol=1e-5, abstol=1e-6)

    if sol.retcode != :Success
        return 9999.0, st, NamedTuple()
    end

    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, length(tsteps))
    pred = permutedims(sol_reshaped, (1, 3, 2))

    # Loss only on observed channel 1
    loss = mean(abs2, data[1:1, :, :] .- pred)
    return loss, st, NamedTuple()
end

# Training Helpers
# Use Lux.Training.TrainState directly

function train_curriculum!(model, ps, st, opt, tsteps, X_data, curriculum_steps, epochs_per_stage, schedule_list)
    # TrainState takes the optimizer RULE and initializes state internally
    tstate = Lux.Training.TrainState(model, ps, st, opt)

    loss_hist = Dict()
    patience = 50
    min_delta = 0.005

    for (i_stage, n_points) in enumerate(curriculum_steps)
        println("\n=== Stage $i_stage: Horizon = $n_points days ===")
        loss_hist[n_points] = []
        schedule = schedule_list[i_stage]
        nepochs = epochs_per_stage[i_stage]

        subset_data = X_data[:, 1:n_points, :]
        subset_t = tsteps[1:n_points]

        best_loss = Inf
        patience_counter = 0

        function loss_f(m, p, s, d)
            loss_curriculum(m, p, s, d, subset_t, covariate_splines)
        end

        for ep in 1:nepochs
            # LR Schedule
            if haskey(schedule, ep)
                new_lr = schedule[ep]
                Optimisers.adjust!(tstate.optimizer_state, new_lr)
                println("LR adjusted to $new_lr")
            end

            grads, l, _, tstate = Training.single_train_step!(
                AutoZygote(), loss_f, subset_data, tstate
            )
            push!(loss_hist[n_points], l)

            if ep % 10 == 0 || ep == 1
                println("Epoch $ep | Loss: $l")
            end

            # Save checkpoints occasionally
            if i_stage > 3 && ep % 50 == 0
                # Plotting logic could go here
            end

            # Early Stopping
            if l < (best_loss - min_delta)
                best_loss = l
                patience_counter = 0
            else
                patience_counter += 1
            end

            if patience_counter >= patience
                println("Early stopping triggered at Epoch $ep")
                break
            end
        end
    end
    return tstate.parameters, loss_hist
end

# -------------------------------------------------------------
# 5. EXECUTION
# -------------------------------------------------------------

println("Starting Curriculum Training...")
@time ps_trained, loss_hist = train_curriculum!(
    gnn, ps, st_gnn, opt,
    tsteps, X_norm,
    curriculum_steps,
    epochs_per_stage,
    lr_schedule
)

# SAVE
save_path = "Resultados/test-8/checkpoints/params_test8_25s.jld2"
println("Saving calibrated parameters to $save_path")
@save save_path ps_trained

# Simple Plot
println("Plotting final debug 25-states...")
# Just plot 3 random states
mkpath("Resultados/test-8/plots")
idxs_to_plot = [1, 10, 25] # FL, VA, NV
# Generate full prediction (400 days)
final_pred = loss_curriculum(gnn, ps_trained, st_gnn, X_norm, tsteps, covariate_splines)[1]
# This returns loss, we need pred function. Re-implement predict locally or trust loss.
# To save time, just save params. Visualizer script will handle plots.

println("Done! Time: ", time() - strt_time)
