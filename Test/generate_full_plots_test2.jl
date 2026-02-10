using ComponentArrays
using Lux
using Random
using Optimization
using OrdinaryDiffEq
using Plots
using Statistics
using JLD2
using SciMLSensitivity
using Zygote
using CSV
using DataFrames
using Interpolations
using LinearAlgebra
using GraphNeuralNetworks
using JSON
using Printf

# Load Preprocessing and Model
include("preprocessing_testing.jl")

# Config
latent_dim = 3
output_dir = "Resultados/test-2/Latent_3/plots"
mkpath(output_dir)

# Aliases
const X_norm = X_total_norm

# Load Model Definition
include("../Train/model_opt.jl")

# Load Parameters
@load "Params/par_opt_new.jld2" ps_trained
ps_full = ps_trained

println("Generating full range predictions (0-401 days)...")
pred_norm = predict(gnn, ps_full, st_gnn, u0, tsteps_test; splines=covariate_splines)

# Denormalize (Log -> Real Cases)
println("Denormalizing predictions (exp(x)-1)...")
pred = exp.(pred_norm) .- 1.0
X_total = exp.(X_total_norm) .- 1.0

states = ["NY", "OH", "NJ", "GA", "IL", "FL", "CA", "VA", "TX", "NC"]
split_day = 180

println("Saving plots to $output_dir...")

for i in 1:length(states)
    state_name = states[i]

    # Data vs Pred
    y_true = X_total[1, :, i]
    y_pred = pred[1, :, i]

    # Plot
    plt = plot(title="Full Prediction: $state_name", xlabel="Days", ylabel="Cases", legend=:topleft)

    # True Data
    scatter!(plt, tsteps_test, y_true, label="Data", color=:blue, markersize=2, alpha=0.6)

    # Prediction
    plot!(plt, tsteps_test, y_pred, label="Model Prediction", linewidth=2, color=:red)

    # Split Line
    vline!(plt, [split_day], label="Train/Test Split", color=:black, linestyle=:dash)

    # Save
    savefig(plt, joinpath(output_dir, "full_pred_$(state_name).png"))
    println("- Plot saved: full_pred_$(state_name).png")
end

println("All plots generated successfully.")
