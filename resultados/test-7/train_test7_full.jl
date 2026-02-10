println(stdout, "--- LOADING LIBRARIES ---")
flush(stdout)



# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays;
using Graphs, Lux, GNNLux;
using DifferentialEquations, DiffEqFlux;
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity;
using LinearAlgebra, Statistics, Random, Plots, CubicSplines;

rng = Random.default_rng()
Random.seed!(rng, 42)

println(stdout, "Running with ", Threads.nthreads(), " threads")
flush(stdout)


# load data, time steps and spline functions for covariates

# Check if preprocessing variables are already loaded (e.g., from Test script)
if !isdefined(Main, :covariate_splines)
    include("../../Train/preprocessing.jl")
end

println(stdout, "Data matrix size: ", size(X_norm))
flush(stdout)

println(stdout, "u0 size: ", size(u0))
flush(stdout)

println(stdout, "Splines matrix size: ", size(covariate_splines))
flush(stdout)


# TEST 7: FULL GRAPH (no modification)
println(stdout, "=== Test 7: FULL GRAPH ===")
flush(stdout)


#################################################
############### NEURAL MODEL ####################
#################################################

# model configuration
latent_dim = 3  # Experiment: Latent 3 (Test 2: Refined)
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes = g.num_nodes

# NN - Increased Width (16 -> 32)
# Explicit Model Definition to prevent GNNChain+Zygote hang
# Frozen Dropout Layer: Keeps mask fixed until explicitly updated
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end

Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
# State holds the mask. Initialized to empty/zeros, will be updated before solve.
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
    if st.mask === nothing
        # Fallback if mask not set (pass through or init zero)
        return x, st
    else
        # Apply pre-computed mask
        return x .* st.mask, st
    end
end

# Explicit Model Definition to prevent GNNChain+Zygote hang
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

# Helper to sample new masks for the batch/trajectory
function sample_dropout_masks(model, st, x_shape)
    # x_shape is (features, nodes)
    # We need to construct a new state 'st' where 'mask' fields are populated

    # Drop 1
    d1 = model.drop1
    mask1 = rand(st.drop1.rng, Float32, (32, x_shape[2])) .> d1.p
    mask1 = mask1 ./ (1 - d1.p)
    st_d1 = (mask=mask1, rng=st.drop1.rng)

    # Drop 2
    d2 = model.drop2
    mask2 = rand(st.drop2.rng, Float32, (32, x_shape[2])) .> d2.p
    mask2 = mask2 ./ (1 - d2.p)
    st_d2 = (mask=mask2, rng=st.drop2.rng)

    # Drop 3
    d3 = model.drop3
    mask3 = rand(st.drop3.rng, Float32, (32, x_shape[2])) .> d3.p
    mask3 = mask3 ./ (1 - d3.p)
    st_d3 = (mask=mask3, rng=st.drop3.rng)

    # Return new FULL state with frozen masks
    return (
        layer1=st.layer1, drop1=st_d1,
        layer2=st.layer2, drop2=st_d2,
        layer3=st.layer3, drop3=st_d3,
        layer4=st.layer4
    )
end

# Usage
gnn = ExplicitGNN(nin_tot, 32, nout, 0.05)
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
    n_nodes = size(data, 3)

    # 1. SAMPLE MASKS ONCE for this integration
    # We need to know the shape of x at the dropout layers.
    # It's always (32, n_nodes) because we fixed hidden dim to 32.
    st_frozen = sample_dropout_masks(model, st, (32, n_nodes))

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

        # Forward pass using FROZEN state
        y, _ = model(g, model_input, p_ode.gnn, st_frozen)

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
    loss = mean(abs2, data[1:1, :, :] .- pred)
    return loss, st, NamedTuple()
end

################################################# 
################ OPTIMIZATION ###################                 
#################################################                 

function predict(model, ps_full, st, u0_obs, tsteps; splines=covariate_splines)
    function dudt(u, p_ode, t)
        # u is the observable state, shape (nin_obs * n_nodes)
        u_reshaped = reshape(u, nin_target, n_nodes)
        # evaluate splines
        t_clamped = clamp(t, tsteps[1], tsteps[end])
        cov_matrix = [s(t_clamped) for s in splines]
        # Get latent features
        latents = p_ode.latent_features # shape (latent_dim, n_nodes)
        # Concatenate 
        model_input = vcat(u_reshaped, cov_matrix, latents) # shape (nin_tot, n_nodes)
        # Forward pass
        y, _ = model(g, model_input, p_ode.gnn, st)
        return vec(y)
    end
    prob = ODEProblem(dudt, vec(u0_obs), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps_full, saveat=tsteps, reltol=1e-5, abstol=1e-6)
    # reshape
    n_times = length(tsteps)
    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, n_times)
    pred = permutedims(sol_reshaped, (1, 3, 2))
    return pred
end

function plot_callback(state_data, pred_data, epoch, loss_history)
    mkpath("Resultados/test-7/plots/training_full")

    # Plot first 6 states as example (individual)
    for node in 1:6
        plt = plot(title="Epoch $(epoch): State $(states[node])", xlabel="Time", ylabel="Log Cases")
        scatter!(plt, tsteps, state_data[1, :, node]; label="Data", color=:orange, markersize=1)
        plot!(plt, tsteps, pred_data[1, :, node]; label="Prediction", linewidth=1)
        savefig(plt, "Resultados/test-7/plots/training_full/epoch_$(epoch)_node_$(node).png")
    end

    # Plot Loss without margin to avoid 'mm' error
    plt_loss = plot(loss_history, title="Loss Evolution (Current Stage)", xlabel="Iteration", ylabel="Loss", legend=false, color=:blue, linewidth=2)
    savefig(plt_loss, "Resultados/test-7/plots/training_full/epoch_$(epoch)_loss.png")
end


function train_curriculum!(model, ps, st, opt, tsteps, X_data, curriculum_steps, epochs_per_stage, schedule_list)
    tstate = Training.TrainState(model, ps, st, opt)
    println(stdout, "Starting curriculum training...")
flush(stdout)

    # initialize loss history
    loss_hist = Dict()

    # parameters for early stopping
    patience = 50 # Increased from 30
    min_delta = 0.005 # Decreased from 0.01

    for (i_stage, n_points) in enumerate(curriculum_steps)
        loss_hist[n_points] = []
        schedule = schedule_list[i_stage]
        nepochs = epochs_per_stage[i_stage]
        println(stdout, "Training on $(n_points) points")
flush(stdout)

        println(stdout, "current LR = $(tstate.optimizer_state.rule.opts[1].eta)")
flush(stdout)

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
                println(stdout, "LR = $(tstate.optimizer_state.rule.opts[1].eta)")
flush(stdout)

            end
            grads, l, _, tstate = Training.single_train_step!(
                AutoZygote(), loss_func, data_subset, tstate
            )
            println(stdout, "Stage $(n_points) | Epoch $(epoch) | Loss = $(l)")
flush(stdout)


            push!(loss_hist[n_points], l)

            # Callback: Save plot every 25 epochs (and first few)
            if epoch % 25 == 0 || epoch < 5
                try
                    # Generate prediction for plotting (no gradient needed)
                    pred = predict(model, tstate.parameters, st, u0, tsteps)
                    plot_callback(X_data, pred, epoch, loss_hist[n_points])
                catch e
                    println(stdout, "Warning: Plotting failed at epoch $epoch: $e")
flush(stdout)

                end
            end

            # Early stopping
            if l < (best_loss_in_stage - min_delta)
                best_loss_in_stage = l
                patience_counter = 0
            else
                patience_counter += 1
            end

            # Check if patience runs out
            if patience_counter >= patience
                println(stdout, "Early stopping triggered!")
flush(stdout)

                break
            end
        end
    end
    return tstate.parameters, loss_hist
end



curriculum_steps = [5, 10, 20, 40, 60, 90, 120, 150, 180] # Smoother transitions
epochs_per_stage = [50, 50, 100, 150, 200, 250, 300, 400, 1500] # More epochs, especially at end

# step scheduler rule epoch => new_lr
# step scheduler rule epoch => new_lr
lr_schedule = [
    Dict(1 => 5e-4), # Stage 1
    Dict(1 => 5e-4), # Stage 2
    Dict(1 => 5e-4), # Stage 3
    Dict(1 => 5e-4), # Stage 4
    Dict(1 => 5e-4), # Stage 5
    Dict(1 => 5e-4), # Stage 6
    Dict(1 => 5e-4), # Stage 7
    Dict(1 => 5e-4), # Stage 8
    Dict(1 => 5e-4, 500 => 1e-5, 1000 => 1e-6) # Stage 9 (1500 eps)
]
opt = Optimisers.AdamW(eta=5e-4, lambda=1.0f-4)


if abspath(PROGRAM_FILE) == @__FILE__
    println(stdout, "Starting Training/Opt...")
flush(stdout)

    @time ps_trained, loss_hist = train_curriculum!(
        gnn, ps, st_gnn, opt,
        tsteps, X_norm,
        curriculum_steps,
        epochs_per_stage,
        lr_schedule
    )

    #################################################
    ################## PLOTTING #####################
    #################################################

    # Final prediction
    pred = predict(gnn, ps_trained, st_gnn, u0, tsteps)

    # Save Params
    println(stdout, "Saving parameters to Params/par_opt_test3.jld2")
flush(stdout)

    @save "Resultados/test-7/checkpoints/params_full.jld2" ps_trained

    # Plot
    mkpath("Resultados/test-7/plots/training_full")
    for node in 1:6
        plt = plot(title="Model Prediction (State $(states[node]))", xlabel="Time", ylabel="Log Cases")
        scatter!(plt, tsteps, X_norm[1, :, node]; label="Data", color=:orange, markersize=2)
        plot!(plt, tsteps, pred[1, :, node]; label="Prediction", linewidth=2, color=:red)
        savefig(plt, "Resultados/test-7/plots/training_full/final_state_$(states[node]).png")
    end
end






