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
# Load Preprocessing (Testing version for full splines)
include("preprocessing_testing.jl")

# Manually set latent_dim to 3 (Overwrites any default if needed, though model_opt sets it too)
latent_dim = 3

# ALIASING for model_opt.jl compatibility
# model_opt.jl expects X_norm, u0 (global) to be defined if it skips preprocessing.jl
const X_norm = X_total_norm # Alias for print statements
# covariate_splines is already defined in preprocessing_testing.jl

# Load Model Definition
include("../Train/model_opt.jl")

# Load Parameters
@load "Params/par_opt_new.jld2" ps_trained
ps_full = ps_trained

println("Debug: u0 size: ", size(u0))
println("Debug: ps_full keys: ", keys(ps_full))
println("Debug: latent_dim in script: ", latent_dim)
println("Debug: latent_features size: ", size(ps_full.latent_features))

# Predict
println("Generating predictions...")
pred = predict(gnn, ps_full, st_gnn, u0, tsteps_test; splines=covariate_splines) # Dims: (1, 401, 10)

# Calculate Metrics
states = ["NY", "OH", "NJ", "GA", "IL", "FL", "CA", "VA", "TX", "NC"]
println("| State | Train MSE (0-180) | Test MSE (180-401) | Train MAE | Test MAE |")
println("|---|---|---|---|---|")

# Indices
idx_train = 1:180
idx_test = 181:401

for i in 1:length(states)
    # Get data and pred for state i
    y_true = X_total_norm[1, :, i]
    y_pred = pred[1, :, i]

    # Split
    train_true = y_true[idx_train]
    train_pred = y_pred[idx_train]

    test_true = y_true[idx_test]
    test_pred = y_pred[idx_test]

    # MSE
    mse_train = mean(abs2, train_true .- train_pred)
    mse_test = mean(abs2, test_true .- test_pred)

    # MAE (Mean Absolute Error)
    mae_train = mean(abs.(train_true .- train_pred))
    mae_test = mean(abs.(test_true .- test_pred))

    # Format output
    @printf("| **%s** | %.5f | %.5f | %.5f | %.5f |\n", states[i], mse_train, mse_test, mae_train, mae_test)
end

# Global Metrics
global_mse_train = mean(abs2, pred[1, idx_train, :] .- X_total_norm[1, idx_train, :])
global_mse_test = mean(abs2, pred[1, idx_test, :] .- X_total_norm[1, idx_test, :])

println("\n### Global Loss Results")
println("- **Final Training Loss (MSE):** $(global_mse_train)")
println("- **Final Test Loss (MSE):** $(global_mse_test)")
