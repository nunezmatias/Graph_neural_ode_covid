#!/usr/bin/env julia
# ==========================================================================
# Test 9 — Graph Characterization & Stratified Holdout Selection
# ==========================================================================
# Loads the pre-computed gravity adjacency matrix, computes 5 topological
# metrics per US state, clusters them via PCA + KMeans, and selects 9
# holdout states that are topologically representative while preserving
# graph connectivity.
#
# Run:  julia --project=. Resultados/test-9/graph_characterization.jl
#
# Outputs (saved to Resultados/test-9/):
#   - metrics_table.csv       Full 49-state metrics DataFrame
#   - pca_clusters.png        2D PCA scatter with clusters & holdout
#   - holdout_selection.json  training_set (40) and holdout_set (9)
# ==========================================================================

using CSV, DataFrames, JSON, Plots, Statistics, LinearAlgebra, Random
using Graphs, SimpleWeightedGraphs

Random.seed!(42)

OUT_DIR = @__DIR__
PROJECT_ROOT = joinpath(OUT_DIR, "..", "..")

# ─────────────────────────────────────────────────────────────────────────
# US Census Regions (for geographic diversity constraint)
# ─────────────────────────────────────────────────────────────────────────
CENSUS_REGIONS = Dict(
    "Northeast" => ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest" => ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South" => ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West" => ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "CA", "OR", "WA"],
)

STATE_TO_REGION = Dict{String,String}()
for (region, sts) in CENSUS_REGIONS
    for s in sts
        STATE_TO_REGION[s] = region
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Load the Gravity Graph
# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("PHASE 1: Loading pre-computed Gravity Graph")
println("="^70)

df_adj = CSV.read(joinpath(PROJECT_ROOT, "Data", "adj_pop_dist.csv"), DataFrame)
states = String.(names(df_adj)[2:end])
N = length(states)
println("  States loaded: $N")

A = Matrix{Float64}(df_adj[:, 2:end])
for i in 1:N
    A[i, i] = 0.0
end  # remove self-loops

# Build SimpleWeightedGraph
g_w = SimpleWeightedGraph(N)
for i in 1:N, j in (i+1):N
    if A[i, j] > 0
        add_edge!(g_w, i, j, A[i, j])
    end
end

println("  Nodes: $(nv(g_w)), Edges: $(ne(g_w))")
println("  Fully connected: $(ne(g_w) == N*(N-1)÷2)")
println()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Feature Extraction (5D Topological Vector)
# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("PHASE 2: Computing Topological Metrics")
println("="^70)

# 1. Weighted Degree: sum of edge weights for each node
w_degree = [sum(A[i, :]) for i in 1:N]

# 2. Weighted Betweenness Centrality
#    Graphs.jl betweenness_centrality works on the unweighted graph.
#    For weighted betweenness we convert weights to "distances" (inverse)
#    and use Graphs.jl with a distance matrix.
#    Strategy: create a distance matrix where d_ij = 1/w_ij (higher weight = shorter path)
dist_matrix = zeros(N, N)
for i in 1:N, j in 1:N
    if i != j && A[i, j] > 0
        dist_matrix[i, j] = 1.0 / A[i, j]
    else
        dist_matrix[i, j] = Inf
    end
end

# Build a weighted graph with inverse weights for shortest-path betweenness
g_dist = SimpleWeightedGraph(N)
for i in 1:N, j in (i+1):N
    if A[i, j] > 0
        add_edge!(g_dist, i, j, 1.0 / A[i, j])
    end
end

w_betweenness = betweenness_centrality(g_dist)

# 3. Eigenvector Centrality (on weighted adjacency)
#    Power iteration on A
function eigenvector_centrality(A; max_iter=1000, tol=1e-8)
    n = size(A, 1)
    x = ones(n) / n
    for _ in 1:max_iter
        x_new = A * x
        norm_x = maximum(abs.(x_new))
        if norm_x == 0
            break
        end
        x_new ./= norm_x
        if maximum(abs.(x_new .- x)) < tol
            return x_new ./ sum(x_new)
        end
        x = x_new
    end
    return x ./ sum(x)
end

w_eigenvector = eigenvector_centrality(A)

# 4. Weighted Clustering Coefficient
#    C_i = (1 / (s_i * (k_i - 1))) * Σ_{j,h} (w_ij * w_ih * w_jh)^(1/3)
#    where s_i = weighted degree, k_i = unweighted degree
function weighted_clustering(A)
    n = size(A, 1)
    cc = zeros(n)
    # normalize weights to [0,1] for the formula
    max_w = maximum(A)
    W = A ./ max_w
    for i in 1:n
        neighbors_i = findall(x -> x > 0, W[i, :])
        ki = length(neighbors_i)
        if ki < 2
            cc[i] = 0.0
            continue
        end
        si = sum(W[i, :])
        tri = 0.0
        for j in neighbors_i, h in neighbors_i
            if j < h && W[j, h] > 0
                tri += (W[i, j] * W[i, h] * W[j, h])^(1 / 3)
            end
        end
        cc[i] = 2.0 * tri / (si * (ki - 1))
    end
    return cc
end

w_clustering = weighted_clustering(A)

# 5. Weighted Closeness Centrality
#    C_close(i) = (N-1) / Σ_j d(i,j)  where d uses inverse weights as distances
#    Note: the graph is fully connected so k-core is useless (all nodes = N-1 core).
#    Closeness centrality discriminates how "nearby" a node is in weighted-path terms.
function weighted_closeness(A)
    n = size(A, 1)
    cc = zeros(n)
    # Build distance matrix (inverse weights)
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j && A[i, j] > 0
            D[i, j] = 1.0 / A[i, j]
        else
            D[i, j] = i == j ? 0.0 : Inf
        end
    end
    # Floyd-Warshall for shortest paths
    for k in 1:n, i in 1:n, j in 1:n
        if D[i, k] + D[k, j] < D[i, j]
            D[i, j] = D[i, k] + D[k, j]
        end
    end
    for i in 1:n
        total_dist = sum(D[i, j] for j in 1:n if j != i)
        cc[i] = total_dist > 0 ? (n - 1) / total_dist : 0.0
    end
    return cc
end

w_closeness = weighted_closeness(A)

# Build DataFrame
metrics = DataFrame(
    state=states,
    weighted_degree=w_degree,
    betweenness=w_betweenness,
    eigenvector=w_eigenvector,
    clustering=w_clustering,
    closeness=w_closeness,
    region=[STATE_TO_REGION[s] for s in states],
)

println(metrics)
println()
println("Descriptive Statistics:")
for col in [:weighted_degree, :betweenness, :eigenvector, :clustering, :closeness]
    vals = metrics[!, col]
    println("  $col: mean=$(round(mean(vals), digits=4)), std=$(round(std(vals), digits=4)), " *
            "min=$(round(minimum(vals), digits=4)), max=$(round(maximum(vals), digits=4))")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Embedding & Clustering (PCA via SVD + KMeans)
# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("PHASE 3: PCA Embedding & KMeans Clustering")
println("="^70)

feature_cols = [:weighted_degree, :betweenness, :eigenvector, :clustering, :closeness]
X = hcat([metrics[!, col] for col in feature_cols]...)  # N × 5

# StandardScaler (Z-score) with zero-variance guard
μ_X = mean(X, dims=1)
σ_X = std(X, dims=1)
# Replace zero std with 1.0 to avoid NaN (constant features become 0 after centering)
σ_X = map(s -> s == 0.0 ? 1.0 : s, σ_X)
X_scaled = (X .- μ_X) ./ σ_X

# PCA via SVD (keep 2 components)
U, S, Vt = svd(X_scaled)
X_pca = X_scaled * Vt[:, 1:2]  # project onto first 2 PCs
explained_var = S .^ 2 ./ sum(S .^ 2)
println("  PCA Explained Variance: PC1=$(round(explained_var[1], digits=3)), " *
        "PC2=$(round(explained_var[2], digits=3)), Total=$(round(sum(explained_var[1:2]), digits=3))")

metrics.PC1 = X_pca[:, 1]
metrics.PC2 = X_pca[:, 2]

# KMeans (manual implementation)
function kmeans_cluster(X, k; max_iter=200, n_init=20, rng=Random.default_rng())
    n, d = size(X)
    best_labels = zeros(Int, n)
    best_inertia = Inf

    for _ in 1:n_init
        # Random initialization
        idx = randperm(rng, n)[1:k]
        centroids = X[idx, :]

        labels = zeros(Int, n)
        for iter in 1:max_iter
            # Assign labels
            for i in 1:n
                dists = [sum((X[i, :] .- centroids[c, :]) .^ 2) for c in 1:k]
                labels[i] = argmin(dists)
            end
            # Update centroids
            new_centroids = zeros(k, d)
            for c in 1:k
                members = findall(==(c), labels)
                if !isempty(members)
                    new_centroids[c, :] = mean(X[members, :], dims=1)
                else
                    new_centroids[c, :] = X[rand(rng, 1:n), :]
                end
            end
            if maximum(abs.(new_centroids .- centroids)) < 1e-8
                centroids = new_centroids
                break
            end
            centroids = new_centroids
        end

        inertia = sum(sum((X[i, :] .- centroids[labels[i], :]) .^ 2) for i in 1:n)
        if inertia < best_inertia
            best_inertia = inertia
            best_labels = copy(labels)
        end
    end

    # Recompute final centroids
    centroids = zeros(k, size(X, 2))
    for c in 1:k
        members = findall(==(c), best_labels)
        if !isempty(members)
            centroids[c, :] = mean(X[members, :], dims=1)
        end
    end

    return best_labels, centroids
end

labels, centroids_scaled = kmeans_cluster(X_scaled, 3; rng=Random.MersenneTwister(42))

# Map centroids back to original scale for interpretation
centroids_orig = centroids_scaled .* σ_X .+ μ_X

# Label clusters semantically: highest degree = Hubs, lowest = Periphery, middle = Connectors
degree_col = 1  # weighted_degree is first column
degree_ranks = centroids_orig[:, degree_col]
hub_idx = argmax(degree_ranks)
periph_idx = argmin(degree_ranks)
conn_idx = setdiff(1:3, [hub_idx, periph_idx])[1]

cluster_names = Dict(hub_idx => "A_Hubs", conn_idx => "B_Connectors", periph_idx => "C_Periphery")
metrics.cluster = labels
metrics.cluster_name = [cluster_names[l] for l in labels]

println("\nCluster Centroids (original scale):")
for (idx, name) in sort(collect(cluster_names), by=x -> x[2])
    vals = round.(centroids_orig[idx, :], digits=4)
    println("  $name: degree=$(vals[1]), btw=$(vals[2]), eig=$(vals[3]), " *
            "clust=$(vals[4]), close=$(vals[5])")
end
println()

for cname in ["A_Hubs", "B_Connectors", "C_Periphery"]
    members = metrics[metrics.cluster_name.==cname, :state]
    println("  $cname ($(length(members))): $members")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Stratified Holdout Selection
# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("PHASE 4: Stratified Holdout Selection (3 per cluster)")
println("="^70)

MAX_REGION_COUNT = 4
MAX_ATTEMPTS = 200

function passes_geo_constraint(holdout_states)
    region_counts = Dict{String,Int}()
    for s in holdout_states
        r = STATE_TO_REGION[s]
        region_counts[r] = get(region_counts, r, 0) + 1
    end
    return all(c <= MAX_REGION_COUNT for c in values(region_counts))
end

function select_holdout(metrics_df, rng)
    holdout = String[]
    for cname in ["A_Hubs", "B_Connectors", "C_Periphery"]
        pool = metrics_df[metrics_df.cluster_name.==cname, :state]
        chosen = pool[randperm(rng, length(pool))[1:min(3, length(pool))]]
        append!(holdout, chosen)
    end
    return holdout
end

rng = Random.MersenneTwister(42)
best_holdout = String[]

for attempt in 1:MAX_ATTEMPTS
    global best_holdout
    candidate = select_holdout(metrics, rng)
    if passes_geo_constraint(candidate)
        best_holdout = candidate
        println("  ✓ Valid holdout found on attempt $attempt")
        break
    end
    if attempt == MAX_ATTEMPTS
        best_holdout = candidate
        println("  ⚠ Could not satisfy geo-constraint in $MAX_ATTEMPTS attempts. Using best available.")
    end
end

println("  Holdout ($(length(best_holdout))): $best_holdout")
region_dist = Dict{String,Int}()
for s in best_holdout
    r = STATE_TO_REGION[s]
    region_dist[r] = get(region_dist, r, 0) + 1
end
println("  Region distribution: $region_dist")
println()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Integrity Check (The Lobotomy Check)
# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("PHASE 5: Integrity Check — Graph Connectivity After Removal")
println("="^70)

FIEDLER_THRESHOLD = 0.01

function compute_fiedler(A_sub)
    # Algebraic connectivity = second smallest eigenvalue of the Laplacian
    n = size(A_sub, 1)
    if n < 2
        return 0.0
    end
    D = diagm(vec(sum(A_sub, dims=2)))
    L = D - A_sub
    eigs = sort(real.(eigvals(L)))
    return eigs[2]
end

function integrity_check!(G_adj, holdout, metrics_df, all_states)
    holdout = copy(holdout)
    train_idx = [i for i in 1:length(all_states) if !(all_states[i] in holdout)]
    A_sub = G_adj[train_idx, train_idx]

    # Check connectivity via graph
    g_sub = SimpleGraph(length(train_idx))
    for i in 1:length(train_idx), j in (i+1):length(train_idx)
        if A_sub[i, j] > 0
            add_edge!(g_sub, i, j)
        end
    end

    is_conn = is_connected(g_sub)
    fiedler = is_conn ? compute_fiedler(A_sub) : 0.0

    println("  G' nodes: $(length(train_idx)), edges: $(ne(g_sub))")
    println("  Connected: $is_conn")
    println("  Fiedler value (algebraic connectivity): $(round(fiedler, digits=6))")

    repair_count = 0
    while (!is_conn || fiedler < FIEDLER_THRESHOLD) && repair_count < 5
        repair_count += 1
        # Find highest betweenness in holdout
        btw_holdout = Dict(s => metrics_df[metrics_df.state.==s, :betweenness][1] for s in holdout)
        bridge_state = sort(collect(btw_holdout), by=x -> x[2], rev=true)[1][1]
        bridge_cluster = metrics_df[metrics_df.state.==bridge_state, :cluster_name][1]

        println("\n  🔧 Repair #$repair_count: Returning '$bridge_state' " *
                "(betweenness=$(round(btw_holdout[bridge_state], digits=4)), " *
                "cluster=$bridge_cluster) to training set")

        filter!(x -> x != bridge_state, holdout)

        # Find replacement from same cluster
        pool = metrics_df[metrics_df.cluster_name.==bridge_cluster, :state]
        pool = [s for s in pool if !(s in holdout) && s != bridge_state]
        sort!(pool, by=s -> metrics_df[metrics_df.state.==s, :betweenness][1])

        replaced = false
        for candidate in pool
            test_holdout = vcat(holdout, [candidate])
            test_train_idx = [i for i in 1:length(all_states) if !(all_states[i] in test_holdout)]
            A_test = G_adj[test_train_idx, test_train_idx]

            g_test = SimpleGraph(length(test_train_idx))
            for i in 1:length(test_train_idx), j in (i+1):length(test_train_idx)
                if A_test[i, j] > 0
                    add_edge!(g_test, i, j)
                end
            end

            if is_connected(g_test)
                test_fiedler = compute_fiedler(A_test)
                if test_fiedler >= FIEDLER_THRESHOLD
                    push!(holdout, candidate)
                    btw_c = metrics_df[metrics_df.state.==candidate, :betweenness][1]
                    println("       Replaced with '$candidate' (betweenness=$(round(btw_c, digits=4)))")
                    replaced = true
                    break
                end
            end
        end

        if !replaced
            println("  ⚠ Could not find valid replacement in cluster $bridge_cluster.")
        end

        # Re-check
        train_idx = [i for i in 1:length(all_states) if !(all_states[i] in holdout)]
        A_sub = G_adj[train_idx, train_idx]
        g_sub = SimpleGraph(length(train_idx))
        for i in 1:length(train_idx), j in (i+1):length(train_idx)
            if A_sub[i, j] > 0
                add_edge!(g_sub, i, j)
            end
        end
        is_conn = is_connected(g_sub)
        fiedler = is_conn ? compute_fiedler(A_sub) : 0.0
        println("  After repair: Connected=$is_conn, Fiedler=$(round(fiedler, digits=6))")
    end

    return holdout, is_conn, fiedler
end

final_holdout, final_connected, final_fiedler = integrity_check!(A, best_holdout, metrics, states)
training_set = sort([s for s in states if !(s in final_holdout)])
holdout_set = sort(final_holdout)

println()
println("  ✅ FINAL RESULT:")
println("     Training set ($(length(training_set))): $training_set")
println("     Holdout set  ($(length(holdout_set))):  $holdout_set")
println("     Connected: $final_connected, Fiedler: $(round(final_fiedler, digits=6))")
println()

# ═══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("Saving outputs...")
println("="^70)

# 1. Metrics table
metrics_path = joinpath(OUT_DIR, "metrics_table.csv")
CSV.write(metrics_path, metrics)
println("  Saved: $metrics_path")

# 2. Holdout selection JSON
selection = Dict(
    "training_set" => training_set,
    "holdout_set" => holdout_set,
    "holdout_clusters" => Dict(s => metrics[metrics.state.==s, :cluster_name][1] for s in holdout_set),
    "integrity" => Dict(
        "connected" => final_connected,
        "fiedler_value" => final_fiedler,
    ),
)
json_path = joinpath(OUT_DIR, "holdout_selection.json")
open(json_path, "w") do f
    JSON.print(f, selection, 2)
end
println("  Saved: $json_path")

# 3. PCA Cluster Plot
colors_map = Dict("A_Hubs" => :red, "B_Connectors" => :orange, "C_Periphery" => :steelblue)
markers_map = Dict("A_Hubs" => :diamond, "B_Connectors" => :rect, "C_Periphery" => :circle)

p = plot(title="Test 9: US State Gravity Graph\nTopological Clustering & Holdout Selection",
    xlabel="PC1 ($(round(explained_var[1]*100, digits=1))% variance)",
    ylabel="PC2 ($(round(explained_var[2]*100, digits=1))% variance)",
    legend=:topright, size=(900, 650), dpi=150, grid=true, gridalpha=0.3)

for cname in ["A_Hubs", "B_Connectors", "C_Periphery"]
    mask = metrics.cluster_name .== cname
    scatter!(p, metrics[mask, :PC1], metrics[mask, :PC2],
        color=colors_map[cname], markershape=markers_map[cname],
        markersize=7, alpha=0.7, markerstrokewidth=0.5,
        label=cname)
end

# Annotate all states
for row in eachrow(metrics)
    annotate!(p, row.PC1, row.PC2 + 0.15, text(row.state, 6, :center, :bold))
end

# Mark holdout states with a large ring
for s in holdout_set
    row = metrics[metrics.state.==s, :]
    scatter!(p, row.PC1, row.PC2, markershape=:circle,
        markersize=14, markercolor=:transparent,
        markerstrokecolor=:red, markerstrokewidth=2.5,
        label=nothing)
end

# Info annotation
info_text = "Holdout: $(join(holdout_set, ", "))\nFiedler: $(round(final_fiedler, digits=4)) | Connected: $final_connected"
annotate!(p, :bottomleft, text(info_text, 7, :left))

plot_path = joinpath(OUT_DIR, "pca_clusters.png")
savefig(p, plot_path)
println("  Saved: $plot_path")

println()
println("Done! 🎯")
