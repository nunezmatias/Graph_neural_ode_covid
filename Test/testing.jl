# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays;
using Graphs, Lux, GNNLux;
using DifferentialEquations, DiffEqFlux;
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity; #OptimizationOptimJL
using LinearAlgebra, Statistics, Random, Plots, Plots.Measures, CubicSplines;

rng = Random.default_rng()
Random.seed!(rng, 42)

println("Running with ", Threads.nthreads(), " threads")

# load data, time steps and spline functions for covariates

include("preprocessing_testing.jl");

println("Data matrix size: ", size(X_total_norm))
println("u0 size: ", size(u0))
println("Spline matrix size: ", size(covariate_splines))

#################################################
############### NEURAL MODEL ####################
#################################################

# model configuration
latent_dim = 4  #latent variables to be optimized
nin_target = 1
nin_covar = 3
nin_tot = nin_target + nin_covar + latent_dim
nout = 1
n_nodes = g.num_nodes

# NN
gnn = GNNLux.GNNChain(GNNLux.GraphConv(nin_tot => 16, tanh; aggr=mean),
    GNNLux.GraphConv(16 => 16, tanh; aggr=mean),
    GNNLux.GraphConv(16 => 16, tanh; aggr=mean),
    GNNLux.GraphConv(16 => nout; aggr=mean))

ps_gnn, st_gnn = Lux.setup(rng, gnn)

# define latent variables
latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes) |> f64

# Combine GNN parameters and latent features into one ComponentArray
ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features) |> f64

#################################################
################## PREDICT ######################
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

    sol = solve(prob, Tsit5(), p=ps_full, saveat=tsteps, reltol=1e-5, abstol=1e-6)

    # reshape
    n_times = length(tsteps)
    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, n_times)
    pred = permutedims(sol_reshaped, (1, 3, 2))

    return pred
end

# load optimized parameters
@load "Params/par_opt_new.jld2" ps_trained # no variable for loss history since I don't have it in this particular file

# Predictions
pred_norm = predict(gnn, ps_trained, st_gnn, u0, tsteps_test)

# denormalize
pred = exp.(pred_norm) .- 1

# Plot
# Plot
mkpath("plots")
for node in 1:n_nodes
    plt = plot(title="State $(states[node]): Cases", xlabel="Time", ylabel="Cases", legend=:topleft, right_margin=15mm)
    # Plot actual data as scatter
    scatter!(plt, tsteps_test, X_total[1, :, node]; label="Data", color=RGB(0, 179 / 255, 60 / 255), markersize=2)
    plot!(plt, tsteps_test, X_total[1, :, node]; label=false, color=RGB(0, 179 / 255, 60 / 255), alpha=0.5, linewidth=1)
    plot!(plt, tsteps_test, pred[1, :, node]; label="Prediciton", color=:crimson, linewidth=1.5)
    vline!([180], linestyle=:dash, color=:black, label="End of training")
    savefig(plt, "plots/state_$(states[node]).png")
    display(plt)
end




