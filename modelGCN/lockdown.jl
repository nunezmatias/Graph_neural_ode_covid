# Make causal analysis

# Import libraries
using Pkg
println("Current working directory: ", pwd())
Pkg.activate("..")

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using LinearAlgebra, Statistics, StatsBase, StatsPlots, Random, Plots, Plots.Measures, CubicSplines
using NPZ, JSON

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

# -------------------------------------------------------------
# 1. Data Loading 
# -------------------------------------------------------------

# Normal
train_states = [
    "AL", "CA", "CO", "CT", "DC", "DE", "FL", "IA", "ID", "IL", "IN", "KS", "KY",
    "LA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "NE", "NH", "NJ", "NM",
    "NY", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WI", "WV", "WY"
]

test_states = ["OH", "GA", "MA", "PA", "AR", "OK", "ND", "VT", "WA", "AZ", "UT", "NV"]

all_states = vcat(train_states, test_states)

# import adjacency matrix 
df_adj_full = CSV.read("../Data/fitted_matrix/flow_mat_exp_gravity_fitted.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
name_to_index = Dict(name => i for (i, name) in enumerate(col_names))
indices = [name_to_index[s] for s in all_states]
adj_raw = Matrix(df_adj_full[:, 2:end])
adj_sub = adj_raw[indices, indices]

function controlled_normalize(A; alpha=0.6f0)
    A_no_diag = copy(A)
    # Remove self-loops entirely for the neighbor calculation
    A_no_diag[diagind(A_no_diag)] .= 0.0
    # Normalize ONLY the neighbor weights
    # matrix has flows origin state on rows, destination on columns, normalize arriving signals on columns to one
    col_sums = sum(A_no_diag, dims=1)
    A_neigh_norm = Float32.(A_no_diag ./ (col_sums .+ 1e-8))
    # Create an Identity matrix
    I_mat = Matrix{Float32}(I, size(A)...)
    # Blend: alpha * Self + (1 - alpha) * Neighbors
    return alpha .* I_mat .+ (1.0f0 - alpha) .* A_neigh_norm
end

A_normal = controlled_normalize(adj_sub; alpha=0.0f0) 

# load data
features_raw = NPZ.npzread("../Data/data_all.npz")  
# Make data tensor
n_nodes = length(all_states)
n_vars, n_times = size(features_raw["NY"])
X_tensor_all = zeros(Float32, n_vars, n_times, n_nodes)
for (i, state) in enumerate(all_states)
    X_tensor_all[:, :, i] = features_raw[state]
end

# select variables
idx_list = [1,2,3,6] 
# Ensure variable 1 (target: daily infections) is always included
if !(1 in idx_list)
    push!(idx_list, 1)
end
idx_list = sort(unique(idx_list)) # Remove duplicates and keep things ordered
X_tensor = X_tensor_all[idx_list,:,:]

# shift variables
lags = [7,7,14]
max_lag = isempty(lags) ? 0 : maximum(lags)
new_T = n_times - max_lag
X_aligned = zeros(eltype(X_tensor), length(idx_list), new_T, n_nodes)
X_aligned[1, :, :] = X_tensor[1, (max_lag + 1):n_times, :]
for i in eachindex(lags)
    l = lags[i]
    var_idx = i + 1 
    start_idx = max_lag - l + 1
    end_idx = n_times - l
    X_aligned[var_idx, :, :] = X_tensor[var_idx, start_idx:end_idx, :]
end

n_vars, n_times, n_nodes = size(X_aligned)
println("Vars, Times, Nodes: ", n_vars, ", ", n_times, ", ", n_nodes)
# Normalize
X_norm = log.(X_aligned .+ 1.0f0)
# Time
tsteps = Float32.(collect(0:n_times-1))

# -------------------------------------------------------------
# 2. Model definition
# -------------------------------------------------------------

# Frozen Dropout Layer
struct FrozenDropout{T} <: Lux.AbstractLuxLayer
    p::T
end

Lux.initialparameters(rng::Random.AbstractRNG, ::FrozenDropout) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, ::FrozenDropout) = (mask=nothing, rng=rng)

function (d::FrozenDropout)(x, ps, st)
    if st.mask === nothing
        return x, st
    else
        return (x .* st.mask) / (1 - d.p), st 
    end
end


# DECOUPLED SPATIAL GRAPH LAYER
struct DecoupledGraphLayer{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    act::F
end

DecoupledGraphLayer(in_dims::Int, out_dims::Int; act=identity) = DecoupledGraphLayer(in_dims, out_dims, act)

function Lux.initialparameters(rng::Random.AbstractRNG, l::DecoupledGraphLayer)
    # W_self transforms state-level internal dynamics
    # W_neigh transforms spatial diffusion inflows from neighbors
    (W_self  = Lux.glorot_uniform(rng, l.out_dims, l.in_dims),
     W_neigh = Lux.glorot_uniform(rng, l.out_dims, l.in_dims),
     b       = zeros(Float32, l.out_dims, 1))
end

Lux.initialstates(::Random.AbstractRNG, ::DecoupledGraphLayer) = NamedTuple()

function (l::DecoupledGraphLayer)(x::AbstractMatrix, A_neigh::AbstractMatrix, ps, st)
    # x:       [features × n_nodes]
    # A_neigh: [n_nodes × n_nodes], Strictly neighbor-only (zero diagonal), normalized
    x_agg = x * A_neigh                  # Spatial aggregation: [features × n_nodes]
    
    # Perform decoupled linear mappings
    out = l.act.(ps.W_self * x .+ ps.W_neigh * x_agg .+ ps.b) 
    return out, st
end

# DECOUPLED ARCHITECTURE 
struct ExplicitGNN{L1,D1,L2,D2,L3,D3,L4} <: Lux.AbstractLuxLayer 
    layer1::L1 
    drop1::D1 
    layer2::L2 
    drop2::D2 
    layer3::L3 
    drop3::D3 
    layer4::L4 
end 

function ExplicitGNN(nin, nhidden, nout, drop_p) 
    return ExplicitGNN( 
        DecoupledGraphLayer(nin, nhidden; act=tanh), 
        FrozenDropout(drop_p), 
        DecoupledGraphLayer(nhidden, nhidden; act=tanh), 
        FrozenDropout(drop_p), 
        DecoupledGraphLayer(nhidden, nhidden; act=tanh), 
        FrozenDropout(drop_p), 
        DecoupledGraphLayer(nhidden, nout) 
    ) 
end 

Lux.initialparameters(rng::Random.AbstractRNG, m::ExplicitGNN) = ( 
    layer1=Lux.initialparameters(rng, m.layer1), drop1=Lux.initialparameters(rng, m.drop1), 
    layer2=Lux.initialparameters(rng, m.layer2), drop2=Lux.initialparameters(rng, m.drop2),
    layer3=Lux.initialparameters(rng, m.layer3), drop3=Lux.initialparameters(rng, m.drop3),
    layer4=Lux.initialparameters(rng, m.layer4) 
) 

Lux.initialstates(rng::Random.AbstractRNG, m::ExplicitGNN) = ( 
    layer1=Lux.initialstates(rng, m.layer1), drop1=Lux.initialstates(rng, m.drop1), 
    layer2=Lux.initialstates(rng, m.layer2), drop2=Lux.initialstates(rng, m.drop2), 
    layer3=Lux.initialstates(rng, m.layer3), drop3=Lux.initialstates(rng, m.drop3), 
    layer4=Lux.initialstates(rng, m.layer4) 
) 

function (layer::ExplicitGNN)(A::AbstractMatrix, x::AbstractMatrix, ps, st) 
    x1, st_c1 = layer.layer1(x, A, ps.layer1, st.layer1) 
    x1, st_d1 = layer.drop1(x1, ps.drop1, st.drop1) 

    x2, st_c2 = layer.layer2(x1, A, ps.layer2, st.layer2) 
    x2, st_d2 = layer.drop2(x2, ps.drop2, st.drop2) 
    x2 = x2 .* x1 

    x3, st_c3 = layer.layer3(x2, A, ps.layer3, st.layer3) 
    x3, st_d3 = layer.drop3(x3, ps.drop3, st.drop3) 
    x3 = x3 .+ x2 

    x4, st_c4 = layer.layer4(x3, A, ps.layer4, st.layer4) 

    new_st = (layer1=st_c1, drop1=st_d1, layer2=st_c2, drop2=st_d2, 
              layer3=st_c3, drop3=st_d3, layer4=st_c4) 
    return x4, new_st 
end 

# -------------------------------------------------------------
# 3. Scenario builder and predictions functions
# -------------------------------------------------------------

smoothstep(x) = x^2 * (3 - 2x)

# Continuous edge scaling factor based on time and lockdown status
function get_isolation_factor(t, is_locked)
    if !is_locked
        return 1.0f0
    end
    
    real_start    = 200.0f0
    real_end      = 300.0f0
    duration_drop = 14.0f0
    duration_rise = 30.0f0
    edge_reduction = 0.1f0  # Drop connectivity to 10% (adjust as needed)
    
    if t < real_start
        return 1.0f0
    elseif t < real_start + duration_drop
        progress = (t - real_start) / duration_drop
        return 1.0f0 - (1.0f0 - edge_reduction) * smoothstep(progress)
    elseif t <= real_end
        return edge_reduction
    elseif t < real_end + duration_rise
        progress = (t - real_end) / duration_rise
        return edge_reduction + (1.0f0 - edge_reduction) * smoothstep(progress)
    else
        return 1.0f0
    end
end

function build_splines(X_data, local_idxs::Vector{Int}, lags; lock=false)
    n_var, n_times, n_nodes = size(X_data)
    splines = Matrix{CubicSpline}(undef, n_var-1, n_nodes)
    for n in 1:n_nodes
        covs = copy(X_data[2:end, :, n])
        if (n in local_idxs) && lock
            real_start    = 200
            real_end      = 300
            duration_drop = 14
            reduction     = 0.3
            duration_rise = 30

            for v in 1:size(covs, 1)
                l        = lags[v]
                start_l  = min(real_start + l, n_times)
                end_drop = min(start_l + duration_drop, n_times)
                end_l    = min(real_end + l, n_times)

                # Gradual drop
                for t in start_l:(end_drop - 1)
                    progress   = (t - start_l + 1) / duration_drop
                    factor     = 1.0 - (1.0 - reduction) * smoothstep(progress)
                    covs[v, t] *= factor
                end
                # Plateau at minimum for the lockdown period
                if end_drop <= end_l
                    covs[v, end_drop:end_l] .*= reduction
                end

                # Gradual reopening
                start_rise = end_l + 1
                end_rise   = min(start_rise + duration_rise - 1, n_times)
                for t in start_rise:end_rise
                    progress   = (t - start_rise + 1) / duration_rise
                    # Scale from `reduction` back up to 1.0
                    factor     = reduction + (1.0 - reduction) * smoothstep(progress)
                    covs[v, t] *= factor
                end

            end
        end
        for v in 1:n_var-1
            splines[v, n] = CubicSpline(tsteps, covs[v, :])
        end
    end
    return splines
end

function predict_base(A_normal, splines_matrix, ps_model, current_latents, current_st)
    n_cov = size(splines_matrix, 1)

    function dudt(u, p, t)
        t_c = clamp(t, 0f0, tsteps[end])

        cov_val    = Float32.([splines_matrix[v, n](t_c) for v in 1:n_cov, n in 1:n_nodes])
        u_reshaped = reshape(u, 1, n_nodes)
        input_mat  = vcat(u_reshaped, cov_val, current_latents)
        
        # Call GNN directly with A_normal matrix
        y, _       = gnn(A_normal, input_mat, ps_model.gnn, current_st)
        return vec(y)
    end

    u0   = X_norm[1, 1, :]
    prob = ODEProblem(dudt, u0, (Float32(tsteps[1]), Float32(tsteps[end])))
    sol  = solve(prob, Tsit5(), saveat=tsteps, reltol=1f-3, abstol=1f-3)
    sol_mat      = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_mat, 1, n_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

function predict(A_normal, splines_matrix, ps_model, current_latents, current_st, group_idxs, lock)
    n_cov = size(splines_matrix, 1)
    node_locked = [n in group_idxs for n in 1:n_nodes]

    function dudt(u, p, t)
        t_c = clamp(t, 0f0, tsteps[end])

        # 1. Update Covariates
        cov_val    = Float32.([splines_matrix[v, n](t_c) for v in 1:n_cov, n in 1:n_nodes])
        u_reshaped = reshape(u, 1, n_nodes)
        input_mat  = vcat(u_reshaped, cov_val, current_latents)
        
        # 2. Dynamic Matrix Generation via Matrix Outer Product
        if lock
            node_factors = Float32[get_isolation_factor(t_c, node_locked[n]) for n in 1:n_nodes]
            
            # (node_factors * node_factors') creates an [n_nodes x n_nodes] matrix 
            # where entry (i,j) equals factor[i] * factor[j]
            A_current = A_normal .* (node_factors * node_factors')
        else
            A_current = A_normal
        end

        # 3. Pass the temporary matrix straight to your custom layer setup
        y, _ = gnn(A_current, input_mat, ps_model.gnn, current_st)
        return vec(y)
    end

    u0   = X_norm[1, 1, :]
    prob = ODEProblem(dudt, u0, (Float32(tsteps[1]), Float32(tsteps[end])))
    sol  = solve(prob, Tsit5(), saveat=tsteps, reltol=1f-3, abstol=1f-3)
    sol_mat      = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_mat, 1, n_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

function simulate_lockdowns_model(target_groups::Vector{Vector{String}}, ps_model, current_latents, current_st, A_normal; 
                                  states=all_states, X=X_norm, lags=lags, num_nodes=n_nodes, num_times=n_times)

    num_groups = length(target_groups)
    preds_group_lock = zeros(num_groups, num_times, num_nodes)

    for (i, group) in enumerate(target_groups)
        group_idxs = [findfirst(==(s), states) for s in group]
        group_idxs = filter(!isnothing, group_idxs)

        splines_lock = build_splines(X, group_idxs, lags; lock=true)
        
        # Call the streamlined matrix prediction function
        pred_lock    = predict(A_normal, splines_lock, ps_model, current_latents, current_st, group_idxs, true)
        preds_group_lock[i, :, :] .= pred_lock[1, :, :]
    end
    return preds_group_lock
end

# Plot lockdown vs baseline
function plot_scenarios(data, pred_base, pred_lock, group_idx, target_groups; denorm=false)
    # Extract the label for the chosen group 
    group_states = target_groups[group_idx]
    name_lock    = join(group_states, "+")   # e.g., "NY+NJ+CT"

    for i in eachindex(all_states)
        state = all_states[i]
        y_label = "Daily Log Cases"

        # UPDATED: Simplified slicing now that ensemble/singleton dimensions are gone
        data_curve      = data[1, :, i]
        pred_base_curve = pred_base[:, i]              # Changed from [1, :, i] to [:, i]
        pred_lock_curve = pred_lock[group_idx, :, i]    # Changed from pred_lock_group[1, :, i]

        if denorm
            data_curve      = exp.(data_curve)      .- 1
            pred_base_curve = exp.(pred_base_curve) .- 1
            pred_lock_curve = exp.(pred_lock_curve) .- 1
            y_label = "Daily Cases"
        end

        # Check if current state is in the lockdown group
        is_locked = state in group_states
        title_str = is_locked ? "$state ★" : "$state"

        p = plot(
            title=title_str,
            xlabel="Days",
            ylabel=y_label,
            legend=:topleft,
            size=(1000, 600),
            dpi=150,
            left_margin=10mm,
            bottom_margin=10mm
        )

        #scatter!(p, tsteps, data_curve, label="Real Data", color=:gray, alpha=0.5, markersize=3, markerstrokewidth=0)
        plot!(p, tsteps, pred_base_curve, label="Baseline", lw=3, color=:blue, alpha=0.9)
        plot!(p, tsteps, pred_lock_curve, label="Lockdown: $name_lock", lw=3, color=is_locked ? :red : :green, linestyle=:solid, alpha=0.9)
        vspan!(p, [200, 300], color=:green, alpha=0.15, label="Lockdown Period")

        mkpath("Plots/dyn_lock/gr$(group_idx)")
        savefig(p, "Plots/dyn_lock/gr$(group_idx)/lock_$state.png")
        display(p)
    end
end

# -------------------------------------------------------------
# 4. Ensemble Mean Lockdowns and Predictions
# -------------------------------------------------------------

param_file = "./Outputs/params.jld2"
println("\n=== Loading model file: $param_file ===")

# Load parameters from the single file
data = JLD2.load(param_file)
param_key = haskey(data, "ps_final") ? "ps_final" : first(keys(data))
ps_current = data[param_key]

target_groups = [
    ["NY", "NJ", "CT", "DE", "MD", "PA"],    # Group 1: North-East
    ["CA", "WA", "OR", "NV"],          # Group 2: West Coast 
    ["TX", "FL", "LA", "MS", "AL"]   # Group 3: South
    ]

num_groups = length(target_groups)
n_latent_samples = 10  

# Initialize arrays to hold the latent samples
pred_base_samples = zeros(n_latent_samples, n_times, n_nodes)
preds_lock_samples = zeros(n_latent_samples, num_groups, n_times, n_nodes)

splines_base = build_splines(X_norm, Int[], lags)  # no lockdown in baseline

# Prepare Model architecture logic 
latent_dim = size(ps_current.latent_features, 1)
n_cov = n_vars - 1
nin_tot = 1 + n_cov + latent_dim
gnn = ExplicitGNN(nin_tot, 64, 1, 0.0)
_, st_gnn = Lux.setup(rng, gnn)

train_idx = [findfirst(==(s), all_states) for s in train_states]
test_idx  = [findfirst(==(s), all_states) for s in test_states]

latents_trained = ps_current.latent_features
μ_lat = mean(latents_trained)
σ_lat = std(latents_trained)

# Simulate scenarios across latent draws
println("Simulating scenarios with $n_latent_samples latent samples...")
for s in 1:n_latent_samples
    # 1. Sample new latents for test states matching the training distribution
    new_latents_test = (randn(rng, latent_dim, length(test_states)) .* σ_lat) .+ μ_lat
    current_latents = Float32.(hcat(latents_trained, new_latents_test))

    # 2. Run baseline prediction (passing ps_current directly since it's node-agnostic)
    pred_base_model = predict_base(A_normal, splines_base, ps_current, current_latents, st_gnn)
    pred_base_samples[s, :, :] .= pred_base_model[1, :, :]

    # 3. Run lockdown simulation scenarios
    preds_lock_model = simulate_lockdowns_model(target_groups, ps_current, current_latents, st_gnn, A_normal)
    preds_lock_samples[s, :, :, :] .= preds_lock_model
end

# Compute Median across the latent sample dimension (dim=1)
pred_base  = dropdims(median(pred_base_samples, dims=1), dims=1)
preds_lock = dropdims(median(preds_lock_samples, dims=1), dims=1)

# Plot group i
for i in [1,2,3]
    plot_scenarios(X_norm, pred_base, preds_lock, i, target_groups; denorm=true)
end

# -------------------------------------------------------------
# 5. Distance effect heatmap
# -------------------------------------------------------------

using PlotlyJS

# Calculate relative difference between baseline and lockdown

# States populations
pop_df = CSV.read("../Data/us_pop_by_state.csv", DataFrame)
pop_df = pop_df[:, ["state_code", "2020_census"]]
pop_df = pop_df[in.(pop_df.state_code, Ref(all_states)), :]
pop_df = pop_df[sortperm(indexin(pop_df.state_code, all_states)), :]
population = pop_df."2020_census"

# lockdown time
t_lock_start = 200
t_lock_end   = 300
lock_range   = t_lock_start+1 : t_lock_end   

# UPDATED COMMENTS: Shapes after removing the ensemble dimension
# pred_base  : shape (n_times, n_nodes)          — median across latent samples
# preds_lock : shape (num_groups, n_times, n_nodes) — median across latent samples

function relative_deviation(base::AbstractArray, lock::AbstractArray, trange)
    b = mean(base[trange])
    l = mean(lock[trange])
    return (l - b) / (abs(b) + 1f-8)   # (lockdown - baseline) / |baseline|
end

rel_dev = zeros(num_groups, n_nodes)

for g in 1:num_groups
    for n in 1:n_nodes
        # UPDATED: pred_base is now 2D, so we drop the first dummy dimension
        base_vec = pred_base[:, n]
        lock_vec = preds_lock[g, :, n]
        rel_dev[g, n] = relative_deviation(base_vec, lock_vec, lock_range)
    end
end

# Build the map

# Codici FIPS 
state_codes = all_states   # "NY", "CA", ...

# Lockdown states specific color
LOCKDOWN_COLOR = "firebrick"

function make_lockdown_map(group_idx::Int; rel_dev_mat=rel_dev, states=all_states, groups=target_groups, clim=(-0.10, -0.00))

    locked_set = Set(groups[group_idx])
    non_locked = [n for n in 1:length(states) if !(states[n] in locked_set)]
    
    # dynamic limits
    vals = rel_dev_mat[group_idx, non_locked]
    #zmin, zmax = minimum(vals) * 100, maximum(vals) * 100

    n = length(states)
    z_vals = Vector{Float32}(undef, n)
    color_vals = Vector{String}(undef, n)
    hover_text = Vector{String}(undef, n)

    for i in 1:n
        s = states[i]
        dev = rel_dev_mat[group_idx, i] * 100   # in %

        if s in locked_set
            z_vals[i] = NaN32         
            color_vals[i] = LOCKDOWN_COLOR
            hover_text[i] = "$s (lockdown)<br>scarto: $(round(dev, digits=1)) %"
        else
            z_vals[i] = Float32(dev)
            hover_text[i] = "$s<br>scarto: $(round(dev, digits=1)) %"
        end
    end

    trace_normal = choropleth(
        locations = states[map(s -> !(s in locked_set), states)],
        z = z_vals[map(s -> !(s in locked_set), states)],
        text = hover_text[map(s -> !(s in locked_set), states)],
        hovertemplate = "%{text}<extra></extra>",
        locationmode = "USA-states",
        colorscale = [[0.0, "#084594"], [0.25, "#2171b5"], [0.5, "#4292c6"], [0.75, "#9ecae1"], [1.0, "#deebf7"]],
        zmin = clim[1] * 100, 
        zmax = clim[2] * 100, 
        colorbar = attr(
            title = attr(text="Relative difference (%)", side="right"),
            thickness = 16,
            len = 0.6,
            ticksuffix  = "%",
        ),
        marker = attr(line=attr(color="white", width=1.0)),
        showscale = true,
    )

    locked_states = collect(locked_set)
    locked_z = zeros(length(locked_states))   # valore fittizio
    locked_hover  = [
        let i = findfirst(==(s), states)
            dev = rel_dev_mat[group_idx, i] * 100
            "$s (lockdown)<br>scarto: $(round(dev, digits=1)) %"
        end
        for s in locked_states
    ]

    trace_locked = choropleth(
        locations = locked_states,
        z = locked_z,
        text = locked_hover,
        hovertemplate = "%{text}<extra></extra>",
        locationmode = "USA-states",
        colorscale = [[0.0, LOCKDOWN_COLOR], [1.0, LOCKDOWN_COLOR]],
        zmin = 0, zmax = 0,
        marker = attr(line=attr(color="white", width=1.5)),
        showscale = false,
    )

    group_label = join(groups[group_idx], " + ")
    layout = Layout(
        title = attr(
            text = "Relative difference lockdown: $group_label",
            font = attr(size=16),
            x  = 0.5,
        ),
        geo = attr(
            scope           = "usa",
            showlakes       = true,
            lakecolor       = "white",
            showland        = true,
            landcolor       = "#F8F8F6",
            showframe       = false,
            coastlinecolor  = "#BBBBBB",
        ),
        paper_bgcolor = "white",
        width  = 900,
        height = 550,
        margin = attr(l=20, r=20, t=60, b=20),

        # Annotazione legenda manuale per il colore lockdown
        annotations = [attr(
            x=0.97, y=0.12,
            xref="paper", yref="paper",
            text="<b style='color:firebrick'>■</b> Lockdown states",
            showarrow=false,
            font=attr(size=12),
            xanchor="left",
        )],
    )

    return PlotlyJS.plot([trace_normal, trace_locked], layout)
end

# Make and save maps for each lockdown group

for g in 1:num_groups
    non_locked = [n for n in 1:n_nodes if !(all_states[n] in target_groups[g])]
    vals = rel_dev[g, non_locked]
    #clim = (minimum(vals), maximum(vals))

    fig = make_lockdown_map(g; rel_dev_mat=rel_dev)
    display(fig)

    PlotlyJS.savefig(fig,"Plots/heatmap_lock_group$(g).png",width=900, height=550)
    println("Map saved for group $g")
end



