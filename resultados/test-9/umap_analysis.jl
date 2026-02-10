#!/usr/bin/env julia
# ==========================================================================
# Test 9 — UMAP Analysis (Multiple Parameter Sets)
# ==========================================================================
# Reads the pre-computed metrics table from graph_characterization.jl,
# runs UMAP with several (n_neighbors, min_dist) combinations on the
# standardized 5D feature space, and produces a multi-panel plot.
#
# Run:  julia --project=. Resultados/test-9/umap_analysis.jl
# ==========================================================================

using CSV, DataFrames, Plots, Statistics, LinearAlgebra, Random, JSON
using UMAP

Random.seed!(42)

OUT_DIR = @__DIR__

# ─────────────────────────────────────────────────────────────────────────
# 1. Load metrics + holdout selection
# ─────────────────────────────────────────────────────────────────────────
println("Loading metrics and holdout selection...")

metrics = CSV.read(joinpath(OUT_DIR, "metrics_table.csv"), DataFrame)
holdout_data = JSON.parsefile(joinpath(OUT_DIR, "holdout_selection.json"))
holdout_set = Set(holdout_data["holdout_set"])

states = metrics.state
N = nrow(metrics)

# Feature matrix (same 5 features as in graph_characterization.jl)
feature_cols = [:weighted_degree, :betweenness, :eigenvector, :clustering, :closeness]
X = hcat([metrics[!, col] for col in feature_cols]...)  # N × 5

# StandardScaler (Z-score) with zero-variance guard
μ_X = mean(X, dims=1)
σ_X = std(X, dims=1)
σ_X = map(s -> s == 0.0 ? 1.0 : s, σ_X)
X_scaled = (X .- μ_X) ./ σ_X

# Transpose: UMAP.jl expects features × samples (5 × N)
X_t = Matrix(X_scaled')

println("  Loaded $N states, $(size(X_t, 1)) features")
println()

# ─────────────────────────────────────────────────────────────────────────
# Helper: extract 2D coordinates from UMAP result
# ─────────────────────────────────────────────────────────────────────────
function extract_coords(result)
    emb = result.embedding  # Vector{Vector{Float64}} of length N
    u1 = [emb[i][1] for i in 1:length(emb)]
    u2 = [emb[i][2] for i in 1:length(emb)]
    return u1, u2
end

# ─────────────────────────────────────────────────────────────────────────
# 2. UMAP Parameter Sweep
# ─────────────────────────────────────────────────────────────────────────

# Parameter grid: (n_neighbors, min_dist)
param_grid = [
    (5, 0.01),
    (5, 0.3),
    (5, 0.8),
    (10, 0.01),
    (10, 0.3),
    (10, 0.8),
    (15, 0.01),
    (15, 0.3),
    (15, 0.8),
]

println("Running UMAP with $(length(param_grid)) parameter combinations...")
println()

# Cluster colors and markers
colors_map = Dict("A_Hubs" => :red, "B_Connectors" => :orange, "C_Periphery" => :steelblue)
markers_map = Dict("A_Hubs" => :diamond, "B_Connectors" => :rect, "C_Periphery" => :circle)

subplots = []

for (nn, md) in param_grid
    println("  n_neighbors=$nn, min_dist=$md ...")

    result = UMAP.fit(X_t, 2; n_neighbors=nn, min_dist=md, n_epochs=500)
    u1, u2 = extract_coords(result)

    p = plot(title="nn=$nn, md=$md", xlabel="UMAP1", ylabel="UMAP2",
        titlefontsize=9, guidefontsize=7, tickfontsize=6,
        legend=false, grid=true, gridalpha=0.3)

    for cname in ["A_Hubs", "B_Connectors", "C_Periphery"]
        mask = metrics.cluster_name .== cname
        scatter!(p, u1[mask], u2[mask],
            color=colors_map[cname], markershape=markers_map[cname],
            markersize=5, alpha=0.7, markerstrokewidth=0.5)
    end

    # Annotate states
    y_range = maximum(u2) - minimum(u2)
    offset = y_range > 0 ? 0.04 * y_range : 0.1
    for i in 1:N
        annotate!(p, u1[i], u2[i] + offset,
            text(states[i], 5, :center, :bold))
    end

    # Mark holdout with red ring
    for i in 1:N
        if states[i] in holdout_set
            scatter!(p, [u1[i]], [u2[i]], markershape=:circle,
                markersize=10, markercolor=:transparent,
                markerstrokecolor=:red, markerstrokewidth=2.0,
                label=nothing)
        end
    end

    push!(subplots, p)
end

# ─────────────────────────────────────────────────────────────────────────
# 3. Compose & Save multi-panel figure
# ─────────────────────────────────────────────────────────────────────────
println()
println("Composing multi-panel figure...")

# 3×3 grid
fig = plot(subplots...,
    layout=(3, 3), size=(1200, 1000), dpi=150,
    plot_title="Test 9: UMAP Parameter Sweep — Topological Embedding\n" *
               "◆ A_Hubs  ■ B_Connectors  ● C_Periphery  (Red ○ = Holdout)",
    plot_titlefontsize=11)

plot_path = joinpath(OUT_DIR, "umap_sweep.png")
savefig(fig, plot_path)
println("  Saved: $plot_path")

# ─────────────────────────────────────────────────────────────────────────
# 4. Standalone best UMAP (n_neighbors=10, min_dist=0.3)
# ─────────────────────────────────────────────────────────────────────────
println("Generating standalone best UMAP plot (nn=10, md=0.3)...")
result_best = UMAP.fit(X_t, 2; n_neighbors=10, min_dist=0.3, n_epochs=500)
u1b, u2b = extract_coords(result_best)

p_best = plot(title="UMAP Embedding (n_neighbors=10, min_dist=0.3)",
    xlabel="UMAP1", ylabel="UMAP2",
    legend=:topright, size=(900, 650), dpi=150,
    grid=true, gridalpha=0.3)

for cname in ["A_Hubs", "B_Connectors", "C_Periphery"]
    mask = metrics.cluster_name .== cname
    scatter!(p_best, u1b[mask], u2b[mask],
        color=colors_map[cname], markershape=markers_map[cname],
        markersize=7, alpha=0.7, markerstrokewidth=0.5,
        label=cname)
end

y_range_b = maximum(u2b) - minimum(u2b)
offset_b = y_range_b > 0 ? 0.04 * y_range_b : 0.1
for i in 1:N
    annotate!(p_best, u1b[i], u2b[i] + offset_b,
        text(states[i], 6, :center, :bold))
end

for i in 1:N
    if states[i] in holdout_set
        scatter!(p_best, [u1b[i]], [u2b[i]],
            markershape=:circle, markersize=14, markercolor=:transparent,
            markerstrokecolor=:red, markerstrokewidth=2.5, label=nothing)
    end
end

info_text = "Holdout: $(join(sort(collect(holdout_set)), ", "))"
annotate!(p_best, :bottomleft, text(info_text, 7, :left))

best_path = joinpath(OUT_DIR, "umap_best.png")
savefig(p_best, best_path)
println("  Saved: $best_path")

println()
println("Done! 🎯")
