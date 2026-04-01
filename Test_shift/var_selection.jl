# Model features selection

# Import libraries
using CSV, NPZ, JSON, DataFrames, SparseArrays, JLD2, ComponentArrays
using LinearAlgebra, Statistics, StatsBase, Random, Plots, StatsPlots
using Clustering

# 1. LOAD DATA
df_adj_full = CSV.read("Data/adj_pop_dist.csv", DataFrame)
all_states = names(df_adj_full)[2:end]
features_raw = NPZ.npzread("Data/data_all.npz") 

# Make data tensor
n_vars, n_times = size(features_raw["NY"])
n_covar = n_vars-1
n_nodes = length(features_raw)
X_tensor_all = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(all_states)
    X_tensor_all[:, :, i] = features_raw[state]
end
Y = X_tensor_all[1:1,:,:] 
X = X_tensor_all[2:end,:,:]

# 2. FLATTEN DATA
X_flat = reshape(X, n_covar, :) 
Y_flat = reshape(Y, :)

# 3. DISTANCE MATRIX 
C = cor(X_flat')
D = 1.0 .- C # Correlation to distance metric (D = 1 - |C|)

# 4. HIERARCHICAL CLUSTERING 
hc = hclust(D, linkage=:average) # average linkage
# Cut the dendrogram. 
distance_threshold = 0.25 # group feature with correlation higher than 1-distance_threshold
cluster_assignments = cutree(hc, h=distance_threshold)
println("Cluster assignments for the 10 covariates: ", cluster_assignments)

covariate_names = ["CLI community", "Doctor visits", "Events indoor", "Google Cough", "Google Fever", "Google Nasal Congestion", "Google Sore Throath", "Mask last 7 days", "Use of public transit", "Work outside"]
plot(hc, 
     xticks=(1:length(covariate_names), covariate_names),
     xrotation=45,
     ylabel="Distance",
     bottom_margin=10Plots.mm)

# 5. SELECT BEST CLUSTER REPRESENTATIVE
# Calculate the absolute correlation of each covariate with the target Y
target_correlations = [abs(cor(X_flat[i, :], Y_flat)) for i in 1:n_covar]
selected_covariates = Int[]
for cluster_id in unique(cluster_assignments)
    # Find which covariates belong to this specific cluster
    covariates_in_cluster = findall(x -> x == cluster_id, cluster_assignments)
    # Find the covariate in this cluster that has the highest correlation with Y
    corrs_in_cluster = target_correlations[covariates_in_cluster]
    best_idx_in_cluster = argmax(corrs_in_cluster)
    # Map back to the original covariate index and save it
    best_covariate = covariates_in_cluster[best_idx_in_cluster]
    push!(selected_covariates, best_covariate)
end

println("Final Selected Covariate Indices: ", sort(selected_covariates))
println("Reduced from ", n_covar, " to ", length(selected_covariates), " features.")

# CHECK LAG EFFECT

function lag_correlation_profile(X, Y, i; maxlag=14)
    n_nodes = size(X, 3)
    T = size(X, 2)

    lags = 0:maxlag
    corrs = zeros(Float32, length(lags))

    for (idx, lag) in enumerate(lags)
        vals = Float32[]

        for j in 1:n_nodes
            x = vec(X[i, :, j])
            y = vec(Y[1, :, j])

            if lag == 0
                c = cor(x, y)
            else
                c = cor(x[1:T-lag], y[1+lag:T])
            end

            push!(vals, c)
        end

        corrs[idx] = mean(vals)  # average across nodes
    end

    return lags, corrs
end

function plot_all_lag_profiles(X, Y, covariates_names; maxlag=30)
    n_covar = size(X, 1)
    p = plot(layout=(2, ceil(Int, n_covar/2)), size=(1400,650))
    for i in 1:n_covar
        lags, corrs = lag_correlation_profile(X, Y, i; maxlag=maxlag)

        max_idx = argmax(abs.(corrs))
        best_corr = corrs[max_idx]
        best_lag = lags[max_idx]
        println("$(covariates_names[i]): Max Correlation = $(round(best_corr, digits=4)) at Lag = $(best_lag)")

        plot!(p[i], lags, corrs,
            title="$(covariates_names[i])",
            marker=:circle,
            xlabel="Lag",
            ylabel="Corr",
            legend=false,
            left_margin=10Plots.mm,
            bottom_margin=10Plots.mm)
    end

    return p
end


plot_all_lag_profiles(X, Y, covariate_names)

