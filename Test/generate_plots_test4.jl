# Import libraries
using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using Optimization, OptimizationOptimisers, Optim, Zygote, SciMLSensitivity
using LinearAlgebra, Statistics, Random, Plots, CubicSplines
using JSON

# 1. SETUP
# ==============================================================================
include("../Train/preprocessing.jl")

# FILTER: Keep only Active Cases (Indices: 1)
# Must match training exactly
X_norm = X_norm[1:1, :, :]
u0 = X_norm[1, 1, :]
n_nodes = g.num_nodes
tsteps = collect(0.0:1.0:400.0) # Forecast usually goes further, but let's do 0-400 like Test 2

# 2. MODEL RECONSTRUCTION
# ==============================================================================
rng = Random.default_rng()
latent_dim = 3
nin_target = 1
nin_covar = 0
nin_tot = nin_target + nin_covar + latent_dim
nout = 1

# Architecture: Width 32
gnn = GNNLux.GNNChain(
    GNNLux.GraphConv(nin_tot => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => 32, gelu; aggr=mean),
    GNNLux.GraphConv(32 => nout; aggr=mean)
)

ps_init, st_gnn = Lux.setup(rng, gnn)

println("Loading parameters from Params/par_opt_no_covars.jld2...")
@load "Params/par_opt_no_covars.jld2" ps_trained

# 3. PREDICTION FUNCTION
# ==============================================================================
function predict_forecast(model, ps_full, st, u0_obs, tsteps_sim)
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, nin_target, n_nodes)
        latents = p_ode.latent_features

        # Concatenate: [Observations, Latents] (No covariates)
        model_input = vcat(u_reshaped, latents)

        # Forward pass
        y, _ = model(g, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0_obs), (tsteps_sim[1], tsteps_sim[end]))
    sol = solve(prob, Tsit5(), p=ps_full, saveat=tsteps_sim, reltol=1e-5, abstol=1e-6)

    if length(sol.t) != length(tsteps_sim)
        println("Warning: Solver failed to reach end time.")
        return nothing
    end

    sol_matrix = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_matrix, nin_target, n_nodes, length(tsteps_sim))
    pred = permutedims(sol_reshaped, (1, 3, 2)) # (Vars, Time, Nodes)
    return pred
end

# 4. GENERATE PLOTS
# ==============================================================================
mkpath("Resultados/test-4/plots")

println("Generating forecast...")
pred_norm = predict_forecast(gnn, ps_trained, st_gnn, u0, tsteps)

# Denormalize
pred_denorm = exp.(pred_norm) .- 1
X_denorm = exp.(X_norm) .- 1

# Plot specific states
states_to_plot = ["OH", "NY", "CA", "NJ", "TX", "FL", "IL", "GA", "NC", "VA"]

for (node_idx, state_name) in enumerate(states)
    if state_name in states_to_plot
        p = plot(title="$state_name: No Covariates Forecast", xlabel="Days", ylabel="Cases", size=(800, 500))

        # Real Data (Scatter)
        real_data = X_denorm[1, :, node_idx]
        scatter!(p, 1:length(real_data), real_data,
            label="Real Data", color=:blue, markersize=3, alpha=0.6)

        # Prediction (Line)
        pred_data = pred_denorm[1, :, node_idx]
        plot!(p, tsteps, pred_data,
            label="Model Prediction (Latent=3, No Covars)", color=:red, linewidth=2)

        # Train/Test Split Line (Approx 180 days)
        vline!([180], linestyle=:dash, color=:black, label="Train End")

        savefig(p, "Resultados/test-4/plots/$(state_name)_forecast_no_covars.png")
    end
end

println("Plots saved to Resultados/test-4/plots/")
