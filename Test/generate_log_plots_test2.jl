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
output_dir = "Resultados/test-2/Latent_3/plots_log"
mkpath(output_dir)

# Aliases
const X_norm = X_total_norm

# Load Model Definition
include("../Train/model_opt.jl")

# Load Parameters
@load "Params/par_opt_new.jld2" ps_trained
ps_full = ps_trained

println("Generating full range predictions (0-401 days) in LOG SCALE...")
# predict returns normalized (log) output by default if we don't denormalize
pred_norm = predict(gnn, ps_full, st_gnn, u0, tsteps_test; splines=covariate_splines)

states = ["NY", "OH", "NJ", "GA", "IL", "FL", "CA", "VA", "TX", "NC"]
split_day = 180

println("Saving LOG plots to $output_dir...")

for i in 1:length(states)
    state_name = states[i]

    # Data vs Pred (Both in Log Scale)
    y_true = X_total_norm[1, :, i] # Normalized data is already Log(x+1)
    y_pred = pred_norm[1, :, i]    # Model output is Log(x+1)

    # Plot
    plt = plot(title="Log Prediction: $state_name", xlabel="Days", ylabel="Log Cases", legend=:topleft)

    # True Data
    scatter!(plt, tsteps_test, y_true, label="Data (Log)", color=:blue, markersize=2, alpha=0.6)

    # Prediction
    plot!(plt, tsteps_test, y_pred, label="Model (Log)", linewidth=2, color=:red)

    # Split Line
    vline!(plt, [split_day], label="Train/Test Split", color=:black, linestyle=:dash)

    # Save
    savefig(plt, joinpath(output_dir, "log_pred_$(state_name).png"))
    println("- Plot saved: log_pred_$(state_name).png")
end

println("All LOG plots generated successfully.")
