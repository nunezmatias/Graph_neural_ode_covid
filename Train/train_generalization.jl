# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using JSON

rng = Random.default_rng()
Random.seed!(rng, 42)

println("Running with ", Threads.nthreads(), " threads")

# 1. LOAD DATA & SPLIT
# ==============================================================================
include("./preprocessing.jl")

println("Full Data Size: ", size(X_norm))

# Define Groups based on README selection
# Train Group (East/Midwest/South): NY(35), FL(2), TX(26), IL(3), GA(22), OH(25), PA(19), NC(9), MI(37), NJ(15)
train_indices = [35, 2, 26, 3, 22, 25, 19, 9, 37, 15]

# Filter Data for Training Group
X_train = X_norm[:, :, train_indices]
u0_train = X_train[1, 1, :]
println("Training Data Size (Group A): ", size(X_train))

# Filter Adjacency for Training Group
A_train = A_hat[train_indices, train_indices] # Use A_hat (full normalized) to slice
g_train = GNNGraph(A_train)
println("Training Graph Nodes: ", g_train.num_nodes)

# Re-create Splines for Group A
tsteps = collect(0.0:1.0:(size(X_train, 2)-1))
train_splines = [CubicSpline(tsteps, @view X_train[v, :, n]) for v in 2:size(X_train, 1), n in 1:size(X_train, 3)]

# 2. MODEL CONFIGURATION
# ==============================================================================
latent_dim = 4 # Use 4 latent vars as in Test 2 (proven to work)
nin_target = 1
nin_covar = 3  # We USE covariates for this experiment
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes_train = g_train.num_nodes

# Architecture: Width 32 (Test 2 Standard)
gnn = GNNLux.GNNChain(
    GNNLux.GraphConv(nin_tot => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => nout; aggr=mean)
)

ps_gnn, st_gnn = Lux.setup(rng, gnn)

# Latent variables are specific to the Training Graph
latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes_train) |> f64

ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features) |> f64

# 3. ODE FUNCTION
# ==============================================================================
function loss_curriculum(model, p, st, data, tsteps_batch, splines_batch, graph_obj)
    u0_batch = data[1, 1, :]
    tspan = (tsteps_batch[1], tsteps_batch[end])

    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, graph_obj.num_nodes)

        # Evaluate covariates
        covar_vals = [s(t) for s in splines_batch]
        covar_reshaped = reshape(covar_vals, nin_covar, graph_obj.num_nodes)

        # Get latents
        latents = p_ode.latent_features

        # Input: [Cases, Covariates, Latents]
        model_input = vcat(u_reshaped, covar_reshaped, latents)

        y, _ = model(graph_obj, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0_batch), tspan)

    sol = solve(prob, Tsit5(), p=p, saveat=tsteps_batch,
        sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()),
        reltol=1e-5, abstol=1e-6)

    if length(sol.t) != length(tsteps_batch)
        return Inf, st, NamedTuple()
    end

    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, graph_obj.num_nodes, length(tsteps_batch))
    pred = permutedims(sol_reshaped, (1, 3, 2))

    loss = sum(abs2, data[1:1, :, :] .- pred)
    return loss, st, NamedTuple()
end

# 4. TRAINING LOOP
# ==============================================================================
function train_curriculum!(model, ps, st, opt, tsteps, X_data, splines_full, graph_obj, curriculum_steps, epochs_per_stage, schedule_list)
    tstate = Training.TrainState(model, ps, st, opt)
    println("Starting Generalization Training (Group A)...")

    loss_hist = Dict()
    patience = 50
    min_delta = 0.001

    for (i_stage, n_points) in enumerate(curriculum_steps)
        loss_hist[n_points] = []
        schedule = schedule_list[i_stage]
        nepochs = epochs_per_stage[i_stage]
        println("\n=== STAGE $(i_stage): Training on $(n_points) points ===")

        data_subset = X_data[:, 1:n_points, :]
        tsteps_subset = tsteps[1:n_points]

        # Slice splines: (Vars, Nodes) - time is handled inside the spline object
        # We don't need to slice the spline *object*, just pass it. 
        # The spline function handles `t`.

        function loss_func(m, p, st_loc, data)
            loss_curriculum(m, p, st_loc, data, tsteps_subset, splines_full, graph_obj)
        end

        best_loss = Inf
        p_count = 0

        for epoch in 1:nepochs
            if haskey(schedule, epoch)
                new_lr = schedule[epoch]
                Optimisers.adjust!(tstate.optimizer_state, new_lr)
                println("--> LR Adjusted to $(tstate.optimizer_state.rule.opts[1].eta)")
            end

            _, l, _, tstate = Training.single_train_step!(
                AutoZygote(), loss_func, data_subset, tstate
            )

            if epoch % 10 == 0 || epoch == 1
                println("Stage $(n_points) | Epoch $(epoch) | Loss = $(l)")
            end

            # Early Stopping
            if l < (best_loss - min_delta)
                best_loss = l
                p_count = 0
            else
                p_count += 1
            end

            if p_count >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
    end
    return tstate.parameters, loss_hist
end

# 5. EXECUTION
# ==============================================================================
# Setup Curriculum (Standard Test 2)
curriculum_steps = [5, 10, 20, 60, 120, 180]
epochs_per_stage = [100, 100, 200, 400, 500, 800] # Standard config
lr_schedule = [
    Dict(1 => 1e-3), Dict(1 => 1e-3), Dict(1 => 1e-3),
    Dict(1 => 1e-3), Dict(1 => 1e-3, 300 => 1e-4), Dict(1 => 1e-3, 400 => 1e-4) # 180
]

opt = Optimisers.AdamW(eta=1e-3, lambda=1e-5)

ps_trained, _ = train_curriculum!(
    gnn, ps, st_gnn, opt,
    tsteps, X_train, train_splines, g_train,
    curriculum_steps, epochs_per_stage, lr_schedule
)

println("Saving trained parameters to Params/par_opt_generalization.jld2")
@save "Params/par_opt_generalization.jld2" ps_trained
