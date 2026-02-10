# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays;
using Graphs, Lux, GNNLux;
using DifferentialEquations, DiffEqFlux;
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

rng = Random.default_rng()
Random.seed!(rng, 42)

println("Running with ", Threads.nthreads(), " threads")

# 1. LOAD DATA
# ==============================================================================
include("./preprocessing.jl");

# FILTER: Keep only Active Cases (Indices: 1)
# Original Data: (Variables, Time, Nodes)
# We discard variables 2, 3, 4 (covariates)
println("Original Data matrix size: ", size(X_norm))
X_norm = X_norm[1:1, :, :]
println("Filtered Data matrix size (No Covariates): ", size(X_norm))

u0 = X_norm[1, 1, :]
println("u0 size: ", size(u0))

# 2. MODEL CONFIGURATION (Test 2 Specs)
# ==============================================================================
latent_dim = 3       # User Request: 3 Latent Variables
nin_target = 1
nin_covar = 0        # No Covariates
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes = g.num_nodes

# Architecture: Width 32 (Test 2)
# 4 -> 32 -> 32 -> 32 -> 1
gnn = GNNLux.GNNChain(
    GNNLux.GraphConv(nin_tot => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => nout; aggr=mean)
)

ps_gnn, st_gnn = Lux.setup(rng, gnn)

# Define latent variables
latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes) |> f64

# Combine parameters
ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features) |> f64

# 3. ODE FUNCTION
# ==============================================================================
function loss_curriculum(model, p, st, data, tsteps, splines)
    u0_data = data[1, 1, :]
    tspan = (tsteps[1], tsteps[end])

    # Build ODEProblem
    function dudt(u, p_ode, t)
        # u is the observable state
        u_reshaped = reshape(u, nin_target, n_nodes)

        # NO Spline Evaluation (No Covariates)

        # Get latent features
        latents = p_ode.latent_features

        # Concatenate: [Observations, Latents]
        model_input = vcat(u_reshaped, latents)

        # Forward pass
        y, _ = model(g, model_input, p_ode.gnn, st)

        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0_data), tspan)

    # Solve
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps,
        sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()),
        reltol=1e-5, abstol=1e-6)

    # Reshaping
    n_times = length(tsteps)
    if length(sol.t) != n_times
        return Inf, st, NamedTuple() # Failed integration punishment
    end

    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, n_times)
    pred = permutedims(sol_reshaped, (1, 3, 2))

    # Loss 
    loss = sum(abs2, data[1:1, :, :] .- pred)
    return loss, st, NamedTuple()
end

# 4. TRAINING LOOP
# ==============================================================================
function train_curriculum!(model, ps, st, opt, tsteps, X_data, curriculum_steps, epochs_per_stage, schedule_list)
    tstate = Training.TrainState(model, ps, st, opt)
    println("Starting curriculum training (NO COVARIATES)...")

    # Initialize loss history
    loss_hist = Dict()

    # Parameters for early stopping (Test 2: patience=50)
    patience = 50
    min_delta = 0.001

    for (i_stage, n_points) in enumerate(curriculum_steps)
        loss_hist[n_points] = []
        schedule = schedule_list[i_stage]
        nepochs = epochs_per_stage[i_stage]
        println("\n=== STAGE $(i_stage): Training on $(n_points) points ===")

        data_subset = X_data[:, 1:n_points, :]
        tsteps_subset = tsteps[1:n_points]

        # Dummy splines argument (unused)
        function loss_func(m, p, st_loc, data)
            loss_curriculum(m, p, st_loc, data, tsteps_subset, [])
        end

        best_loss_in_stage = Inf
        patience_counter = 0

        for epoch in 1:nepochs
            # LR Schedule Update
            if haskey(schedule, epoch)
                new_lr = schedule[epoch]
                Optimisers.adjust!(tstate.optimizer_state, new_lr)
                println("--> LR Adjusted to $(tstate.optimizer_state.rule.opts[1].eta)")
            end

            grads, l, _, tstate = Training.single_train_step!(
                AutoZygote(), loss_func, data_subset, tstate
            )

            # Print Loss
            println("Stage $(n_points) | Epoch $(epoch) | Loss = $(l)")
            push!(loss_hist[n_points], l)

            # Early stopping check
            if l < (best_loss_in_stage - min_delta)
                best_loss_in_stage = l
                patience_counter = 0
            else
                patience_counter += 1
            end

            if patience_counter >= patience
                println("Early stopping triggered at epoch $epoch")
                break
            end
        end
    end
    return tstate.parameters, loss_hist
end

# 5. EXECUTION CONFIG (Test 2 Curriculum)
# ==============================================================================
# Stages: 9 steps smooth progression
curriculum_steps = [5, 10, 20, 40, 60, 90, 120, 150, 180]

# Epochs per stage (Doubled as per user request)
epochs_per_stage = [200, 200, 400, 400, 600, 800, 1000, 1600, 3000]

# LR Schedule (Aggressive decay in later stages)
lr_schedule = [
    Dict(1 => 1e-3),                    # 5 pts
    Dict(1 => 1e-3),                    # 10 pts
    Dict(1 => 1e-3),                    # 20 pts
    Dict(1 => 1e-3),                    # 40 pts
    Dict(1 => 1e-3),                    # 60 pts
    Dict(1 => 1e-3),                    # 90 pts
    Dict(1 => 1e-3, 300 => 1e-4),       # 120 pts
    Dict(1 => 1e-3, 400 => 1e-4),       # 150 pts
    Dict(1 => 1e-3, 500 => 1e-4, 1000 => 1e-5) # 180 pts (Aggressive)
]

opt = Optimisers.AdamW(eta=1e-3, lambda=0.0)

@time ps_trained, loss_hist = train_curriculum!(
    gnn, ps, st_gnn, opt,
    tsteps, X_norm,
    curriculum_steps,
    epochs_per_stage,
    lr_schedule
);

# 6. SAVE
# ==============================================================================
println("Saving parameters to Params/par_opt_no_covars.jld2")
@save "Params/par_opt_no_covars.jld2" ps_trained
println("Training Complete.")
