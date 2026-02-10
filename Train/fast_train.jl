# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays;
using Graphs, Lux, GNNLux;
using DifferentialEquations, DiffEqFlux;
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

rng = Random.default_rng()
Random.seed!(rng, 42)

println("Running with ", Threads.nthreads(), " threads")

# load data, time steps and spline functions for covariates

include("./preprocessing.jl");

println("Data matrix size: ", size(X_norm))
println("u0 size: ", size(u0))
println("Splines matrix size: ", size(covariate_splines))

#################################################
############### NEURAL MODEL ####################
#################################################

# model configuration
latent_dim = 2  #latent variables to be optimized
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes = g.num_nodes

# NN
gnn = GNNLux.GNNChain(GNNLux.GraphConv(nin_tot => 16, gelu; aggr=mean),
    GNNLux.GraphConv(16 => 16, gelu; aggr=mean),
    GNNLux.GraphConv(16 => 16, gelu; aggr=mean),
    GNNLux.GraphConv(16 => nout; aggr=mean))

ps_gnn, st_gnn = Lux.setup(rng, gnn)

# define latent variables
latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes) |> f64

# Combine GNN parameters and latent features into one ComponentArray
ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features) |> f64

#############################################################
############## CURRICULUM LEARNING SETUP ####################
#############################################################

# ODE constructor and loss

function loss_curriculum(model, p, st, data, tsteps, splines)
    u0_data = data[1, 1, :]
    tspan = (tsteps[1], tsteps[end])

    # Build ODEProblem
    function dudt(u, p_ode, t)
        # u is the observable state, shape (nin_obs * n_nodes)
        u_reshaped = reshape(u, nin_target, n_nodes)
        # evaluate splines
        cov_matrix = map(s -> s(t), splines)
        # Get latent features
        latents = p_ode.latent_features # shape (latent_dim, n_nodes)
        # Concatenate 
        model_input = vcat(u_reshaped, cov_matrix, latents) # shape (nin_tot, n_nodes)
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
    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, n_times)
    pred = permutedims(sol_reshaped, (1, 3, 2))

    # Loss 
    loss = sum(abs2, data[1:1, :, :] .- pred)
    return loss, st, NamedTuple()
end

################################################# 
################ OPTIMIZATION ###################                 
#################################################                 

function predict(model, ps_full, st, u0_obs, tsteps)
    function dudt(u, p_ode, t)
        # u is the observable state, shape (nin_obs * n_nodes)
        u_reshaped = reshape(u, nin_target, n_nodes)
        # evaluate splines
        cov_matrix = [s(t) for s in covariate_splines]
        # Get latent features
        latents = p_ode.latent_features # shape (latent_dim, n_nodes)
        # Concatenate 
        model_input = vcat(u_reshaped, cov_matrix, latents) # shape (nin_tot, n_nodes)
        # Forward pass
        y, _ = model(g, model_input, p_ode.gnn, st)
        return vec(y)
    end
    prob = ODEProblem(dudt, vec(u0_obs), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps_full, saveat=tsteps, reltol=1e-3, abstol=1e-3)
    # reshape
    n_times = length(tsteps)
    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, n_times)
    pred = permutedims(sol_reshaped, (1, 3, 2))
    return pred
end

function plot_callback(state_data, pred_data, epoch)
    mkpath("plots/training")
    # Plot first 3 states as example
    for node in 1:3
        plt = plot(title="Epoch $(epoch): State $(states[node])", xlabel="Time", ylabel="Log Cases")
        scatter!(plt, tsteps, state_data[1, :, node]; label="Data", color=:orange, markersize=1)
        plot!(plt, tsteps, pred_data[1, :, node]; label="Prediction", linewidth=1)
        savefig(plt, "plots/training/epoch_$(epoch)_node_$(node).png")
    end
end


function train_curriculum!(model, ps, st, opt, tsteps, X_data, curriculum_steps, epochs_per_stage, schedule_list)
    tstate = Training.TrainState(model, ps, st, opt)
    println("Starting curriculum training...")
    # initialize loss history
    loss_hist = Dict()

    # parameters for early stopping
    patience = 30
    min_delta = 0.01

    for (i_stage, n_points) in enumerate(curriculum_steps)
        loss_hist[n_points] = []
        schedule = schedule_list[i_stage]
        nepochs = epochs_per_stage[i_stage]
        println("Training on $(n_points) points")
        println("current LR = $(tstate.optimizer_state.rule.opts[1].eta)")
        data_subset = X_data[:, 1:n_points, :]
        tsteps_subset = tsteps[1:n_points]
        function loss_func(m, p, st_loc, data)
            loss_curriculum(m, p, st_loc, data, tsteps_subset, covariate_splines)
        end

        best_loss_in_stage = Inf
        patience_counter = 0

        for epoch in 1:nepochs
            if haskey(schedule, epoch)
                new_lr = schedule[epoch]
                Optimisers.adjust!(tstate.optimizer_state, new_lr)
                println("LR = $(tstate.optimizer_state.rule.opts[1].eta)")
            end
            grads, l, _, tstate = Training.single_train_step!(
                AutoZygote(), loss_func, data_subset, tstate
            )
            println("Stage $(n_points) | Epoch $(epoch) | Loss = $(l)")

            # Callback: Save plot every 50 epochs
            if epoch % 50 == 0
                # Generate prediction for plotting (no gradient needed)
                pred = predict(model, tstate.parameters, st, u0, tsteps)
                plot_callback(X_data, pred, epoch)
            end
            push!(loss_hist[n_points], l)

            # Early stopping
            if l < (best_loss_in_stage - min_delta)
                best_loss_in_stage = l
                patience_counter = 0
            else
                patience_counter += 1
            end

            # Check if patience runs out
            if patience_counter >= patience
                println("Early stopping triggered!")
                break
            end
        end
    end
    return tstate.parameters, loss_hist
end



curriculum_steps = [5, 20, 60, 100, 140, 180]
epochs_per_stage = [2, 2, 2, 2, 2, 2]

# step scheduler rule epoch => new_lr
lr_schedule = [Dict(1 => 1e-3), Dict(1 => 1e-3),
    Dict(1 => 1e-3, 500 => 1e-4), Dict(1 => 1e-3, 800 => 1e-4),
    Dict(1 => 1e-3, 1500 => 1e-4), Dict(1 => 1e-3, 1500 => 1e-4),
    Dict(1 => 1e-3, 2500 => 1e-4), Dict(1 => 1e-3, 2500 => 1e-4),
    Dict(1 => 1e-3, 3500 => 1e-4), Dict(1 => 1e-3, 3500 => 1e-4),
    Dict(1 => 1e-3, 4000 => 1e-4)
]
opt = Optimisers.AdamW(eta=1e-3, lambda=0.0)

@time ps_trained, loss_hist = train_curriculum!(
    gnn, ps, st_gnn, opt,
    tsteps, X_norm,
    curriculum_steps,
    epochs_per_stage,
    lr_schedule
);


#################################################
################## PLOTTING #####################
#################################################



# Get the true initial condition for the observables 

u0_obs_true = X_norm[1, 1, :]

# prediction
pred_norm = predict(gnn, ps_trained, st_gnn, u0_obs_true, tsteps)

# denormalize
# denormalize
pred = exp.(pred_norm) .- 1

# Save trained parameters
println("Saving parameters to Params/par_opt_new.jld2")
@save "Params/par_opt_new.jld2" ps_trained

# Plot
for node in 1:N
    plt = plot(title="State $(states[node]): Cases", xlabel="Time", ylabel="Value")
    # Plot actual data as scatter
    scatter!(plt, tsteps, X_true[1, :, node]; label="Data", color=:orange, markersize=1)
    plot!(plt, tsteps, pred[1, :, node]; label="Prediciton", linewidth=1)
    display(plt)
end






