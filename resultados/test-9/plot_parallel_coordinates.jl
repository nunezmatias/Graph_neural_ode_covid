#!/usr/bin/env julia
# ==========================================================================
# Test 9 — Parallel Coordinates Plot for Cluster Analysis
# ==========================================================================
# Generates a parallel coordinates plot to visualize how the three topological
# clusters (Hubs, Connectors, Periphery) differ across the 5 metrics.
#
# Run:  julia --project=. Resultados/test-9/plot_parallel_coordinates.jl
# ==========================================================================

using CSV, DataFrames, Plots, Statistics

OUT_DIR = @__DIR__
metrics = CSV.read(joinpath(OUT_DIR, "metrics_table.csv"), DataFrame)

# 1. Normalize metrics (Min-Max scaling to [0, 1]) for comparison
feature_cols = [:weighted_degree, :betweenness, :eigenvector, :clustering, :closeness]
feature_names = ["Degree", "Betweenness", "Eigenvector", "Clustering", "Closeness"]

# Create a normalized copy
metrics_norm = copy(metrics)
for col in feature_cols
    min_val = minimum(metrics[!, col])
    max_val = maximum(metrics[!, col])
    metrics_norm[!, col] = (metrics[!, col] .- min_val) ./ (max_val - min_val)
end

# 2. Prepare data for Parallel Coordinates
# We need to construct a series of lines, one per state.

# Define colors for clusters
colors_map = Dict(
    "A_Hubs" => RGBA(0.902, 0.224, 0.275, 0.6),      # vivid red
    "B_Connectors" => RGBA(1.0, 0.647, 0.0, 0.6),    # orange
    "C_Periphery" => RGBA(0.275, 0.510, 0.706, 0.6), # steelblue
)

p = plot(
    title="Cluster Signatures: Parallel Coordinates (Min-Max Scaled)",
    legend=:topright,
    size=(1000, 600),
    dpi=200,
    xticks=(1:length(feature_cols), feature_names),
    xlabel="", ylabel="Normalized Value [0, 1]",
    grid=:y,
    framestyle=:box,
    margin=10Plots.mm
)

# Plot lines for each cluster
# We group by cluster to add a single legend entry per cluster
possible_clusters = ["A_Hubs", "B_Connectors", "C_Periphery"]

for cname in possible_clusters
    cluster_data = metrics_norm[metrics_norm.cluster_name.==cname, :]

    # Extract the feature matrix for this cluster
    # Rows: states, Cols: features
    data_matrix = Matrix(cluster_data[:, feature_cols])

    # We plot all lines at once for efficiency if possible, or loop
    n_states = size(data_matrix, 1)

    if n_states > 0
        # Plot the first one with label, others without
        vals = data_matrix[1, :]
        plot!(p, vals, label=cname, color=colors_map[cname], lw=1.5)

        for i in 2:n_states
            vals = data_matrix[i, :]
            plot!(p, vals, label="", color=colors_map[cname], lw=1.5)
        end
    end
end

# Add some guide lines for 0.0 and 1.0? Not strictly necessary with box frame.

plot_path = joinpath(OUT_DIR, "parallel_coordinates.png")
savefig(p, plot_path)
println("Saved parallel coordinates plot to: $plot_path")
