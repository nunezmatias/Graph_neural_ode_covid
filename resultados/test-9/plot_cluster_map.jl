#!/usr/bin/env julia
# ==========================================================================
# Test 9 — US Geographic Cluster Map
# ==========================================================================
# Generates a choropleth map of the continental US with each state colored
# by its topological cluster (A_Hubs, B_Connectors, C_Periphery).
#
# Uses the same color scheme as umap_best.png.
# Holdout states are marked with a thicker black border.
#
# Run:  julia --project=. Resultados/test-9/plot_cluster_map.jl
# ==========================================================================

using CSV, DataFrames, JSON, Plots, Downloads, Statistics

OUT_DIR = @__DIR__

# ─────────────────────────────────────────────────────────────────────────
# 1. Load cluster assignments
# ─────────────────────────────────────────────────────────────────────────
println("Loading metrics and holdout data...")

metrics = CSV.read(joinpath(OUT_DIR, "metrics_table.csv"), DataFrame)
holdout_data = JSON.parsefile(joinpath(OUT_DIR, "holdout_selection.json"))
holdout_set = Set(holdout_data["holdout_set"])

# Cluster color scheme (matches umap_best.png)
colors_map = Dict(
    "A_Hubs" => RGB(0.902, 0.224, 0.275),  # vivid red
    "B_Connectors" => RGB(1.0, 0.647, 0.0),    # orange
    "C_Periphery" => RGB(0.275, 0.510, 0.706),  # steelblue
)

# Build state → cluster lookup
state_cluster = Dict{String,String}()
for row in eachrow(metrics)
    state_cluster[row.state] = row.cluster_name
end

println("  $(nrow(metrics)) states loaded")
println("  Holdout: $(sort(collect(holdout_set)))")
println()

# ─────────────────────────────────────────────────────────────────────────
# 2. Download & parse US states GeoJSON
# ─────────────────────────────────────────────────────────────────────────
println("Downloading US states GeoJSON...")

geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
geojson_path = joinpath(OUT_DIR, "us-states.json")

if !isfile(geojson_path)
    Downloads.download(geojson_url, geojson_path)
    println("  Downloaded to $geojson_path")
else
    println("  Using cached $geojson_path")
end

geo = JSON.parsefile(geojson_path)
features = geo["features"]
println("  $(length(features)) features found")
println()

# ─────────────────────────────────────────────────────────────────────────
# State name → 2-letter abbreviation mapping
# ─────────────────────────────────────────────────────────────────────────
NAME_TO_CODE = Dict(
    "Alabama" => "AL", "Arizona" => "AZ", "Arkansas" => "AR",
    "California" => "CA", "Colorado" => "CO", "Connecticut" => "CT",
    "Delaware" => "DE", "Florida" => "FL", "Georgia" => "GA",
    "Idaho" => "ID", "Illinois" => "IL", "Indiana" => "IN",
    "Iowa" => "IA", "Kansas" => "KS", "Kentucky" => "KY",
    "Louisiana" => "LA", "Maine" => "ME", "Maryland" => "MD",
    "Massachusetts" => "MA", "Michigan" => "MI", "Minnesota" => "MN",
    "Mississippi" => "MS", "Missouri" => "MO", "Montana" => "MT",
    "Nebraska" => "NE", "Nevada" => "NV", "New Hampshire" => "NH",
    "New Jersey" => "NJ", "New Mexico" => "NM", "New York" => "NY",
    "North Carolina" => "NC", "North Dakota" => "ND", "Ohio" => "OH",
    "Oklahoma" => "OK", "Oregon" => "OR", "Pennsylvania" => "PA",
    "Rhode Island" => "RI", "South Carolina" => "SC", "South Dakota" => "SD",
    "Tennessee" => "TN", "Texas" => "TX", "Utah" => "UT",
    "Vermont" => "VT", "Virginia" => "VA", "Washington" => "WA",
    "West Virginia" => "WV", "Wisconsin" => "WI", "Wyoming" => "WY",
    "District of Columbia" => "DC",
)

# ─────────────────────────────────────────────────────────────────────────
# 3. Plot the map
# ─────────────────────────────────────────────────────────────────────────
println("Generating US cluster map...")

p = plot(
    title="Test 9: US States — Topological Cluster Map\n◆ A_Hubs  ■ B_Connectors  ● C_Periphery",
    xlabel="Longitude", ylabel="Latitude",
    legend=:bottomright, size=(1100, 700), dpi=200,
    aspect_ratio=1.5,
    grid=false,
    background_color=:white,
    titlefontsize=11,
    xlims=(-130, -65), ylims=(24, 50),
)

# Track which clusters we've already added to legend
legend_added = Set{String}()

for feat in features
    state_name = feat["properties"]["name"]
    code = get(NAME_TO_CODE, state_name, nothing)

    # Skip non-continental (Alaska, Hawaii) and states not in our dataset
    if code === nothing || !haskey(state_cluster, code)
        continue
    end

    cluster = state_cluster[code]
    fill_color = colors_map[cluster]
    # Border style: uniform for all states
    border_color = RGBA(0.3, 0.3, 0.3, 0.6)
    border_width = 0.8

    # Label for legend (only first occurrence per cluster)
    lbl = cluster in legend_added ? nothing : cluster
    if lbl !== nothing
        push!(legend_added, cluster)
    end

    geom = feat["geometry"]
    geom_type = geom["type"]
    coords_list = geom["coordinates"]

    # Normalize: MultiPolygon has an extra nesting level
    polygons = geom_type == "MultiPolygon" ? coords_list : [coords_list]

    for poly in polygons
        # poly[1] = exterior ring (list of [lon, lat] pairs)
        ring = poly[1]
        lons = [pt[1] for pt in ring]
        lats = [pt[2] for pt in ring]

        # Filter to continental US viewport
        if maximum(lons) < -130 || minimum(lons) > -65
            continue
        end
        if maximum(lats) < 24 || minimum(lats) > 50
            continue
        end

        plot!(p, Shape(lons, lats),
            fillcolor=fill_color, fillalpha=0.85,
            linecolor=border_color, linewidth=border_width,
            label=lbl)

        # Only label the first polygon of this state
        lbl = nothing
    end
end

# Add state abbreviation labels at polygon centroids
for feat in features
    state_name = feat["properties"]["name"]
    code = get(NAME_TO_CODE, state_name, nothing)
    if code === nothing || !haskey(state_cluster, code)
        continue
    end

    geom = feat["geometry"]
    coords_list = geom["coordinates"]
    polygons = geom["type"] == "MultiPolygon" ? coords_list : [coords_list]

    # Use the largest polygon for centroid calculation
    best_ring = polygons[1][1]
    best_area = 0.0
    for poly in polygons
        ring = poly[1]
        # Approximate area using shoelace
        n = length(ring)
        area = abs(sum(ring[i][1] * ring[mod1(i + 1, n)][2] -
                       ring[mod1(i + 1, n)][1] * ring[i][2] for i in 1:n)) / 2.0
        if area > best_area
            best_area = area
            best_ring = ring
        end
    end

    cx = mean([pt[1] for pt in best_ring])
    cy = mean([pt[2] for pt in best_ring])

    # Skip if outside viewport
    if cx < -130 || cx > -65 || cy < 24 || cy > 50
        continue
    end

    annotate!(p, cx, cy, text(code, 7, :center, :bold, :black))
end

# Info text
# Info text removed as requested

map_path = joinpath(OUT_DIR, "cluster_map_us.png")
savefig(p, map_path)
println("  Saved: $map_path")

println()
println("Done! 🗺️")
