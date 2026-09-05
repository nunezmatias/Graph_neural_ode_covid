# Make causal analysis systematic for all states (Ensemble Mean)

# Import libraries
using Pkg
println("Current working directory: ", pwd())
Pkg.activate("..")

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using LinearAlgebra, Statistics, StatsBase, StatsPlots, Random, Plots, Plots.Measures, CubicSplines
using Plots.Measures
using NPZ, JSON

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

# Create save paths
lock_path = "./Plots/dyn_lock/mlp_temporal"
mkdir(lock_path)
for i in 1:3
    mkdir(joinpath(lock_path, "gr$(i)"))
end
errors_path = "./Plots/errors/mlp_temporal"
mkdir(errors_path)

# -------------------------------------------------------------
# 1. Data Loading 
# -------------------------------------------------------------

all_states = [
    "AL", "CA", "CO", "CT", "DC", "DE", "FL", "IA", "ID", "IL", "IN", "KS", "KY",
    "LA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "NE", "NH", "NJ", "NM",
    "NY", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WI", "WV", "WY",
    "OH", "GA", "MA", "PA", "AR", "OK", "ND", "VT", "WA", "AZ", "UT", "NV"
]

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

A_normal = controlled_normalize(adj_sub; alpha=0.5f0)

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

struct WeightedGraphLayer{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    act::F
end

WeightedGraphLayer(in_dims::Int, out_dims::Int; act=identity) = 
    WeightedGraphLayer(in_dims, out_dims, act)

function Lux.initialparameters(rng::Random.AbstractRNG, l::WeightedGraphLayer)
    (W = Lux.glorot_uniform(rng, l.out_dims, l.in_dims),
     b = zeros(Float32, l.out_dims, 1))
end

Lux.initialstates(::Random.AbstractRNG, ::WeightedGraphLayer) = NamedTuple()

function (l::WeightedGraphLayer)(x::AbstractMatrix, A::AbstractMatrix, ps, st)
    x_agg = x * A                        
    out = l.act.(ps.W * x_agg .+ ps.b)   
    return out, st
end

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
        Lux.Dense(nin, nhidden, tanh),                  # MLP Input layer
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nhidden; act=tanh), # GNN layer 1
        FrozenDropout(drop_p),
        WeightedGraphLayer(nhidden, nhidden; act=tanh), # GNN layer 2
        FrozenDropout(drop_p),
        Lux.Dense(nhidden, nout)                        # MLP Output layer
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

# Forward pass
function (layer::ExplicitGNN)(A::AbstractMatrix, x::AbstractMatrix, ps, st)
    x, st_c1 = layer.layer1(x, ps.layer1, st.layer1)
    x, st_d1 = layer.drop1(x, ps.drop1, st.drop1)
    x, st_c2 = layer.layer2(x, A, ps.layer2, st.layer2)
    x, st_d2 = layer.drop2(x, ps.drop2, st.drop2)
    x, st_c3 = layer.layer3(x, A, ps.layer3, st.layer3)
    x, st_d3 = layer.drop3(x, ps.drop3, st.drop3)
    x, st_c4 = layer.layer4(x, ps.layer4, st.layer4)

    new_st = (layer1=st_c1, drop1=st_d1, layer2=st_c2, drop2=st_d2, 
              layer3=st_c3, drop3=st_d3, layer4=st_c4)
    return x, new_st
end

# -------------------------------------------------------------
# 3. Scenario builder and predictions functions
# -------------------------------------------------------------

# Function to gradually decrease covariates signals and interstate opening during lockdown period
smoothstep(x) = x^2 * (3 - 2x)

# Continuous edge scaling factor based on time and lockdown status
function get_isolation_factor(t, is_locked)
    if !is_locked
        return 1.0f0
    end
    
    real_start    = 200.0f0
    real_end      = 300.0f0
    duration_drop = 30.0f0
    duration_rise = 30.0f0
    edge_reduction = 0.05f0  # Drop connectivity 
    
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

# Build splines for decreased covariates during lockdown
function build_splines(X_data, local_idxs::Vector{Int}, lags; lock=false)
    n_var, n_times, n_nodes = size(X_data)
    splines = Matrix{CubicSpline}(undef, n_var-1, n_nodes)
    for n in 1:n_nodes
        covs = copy(X_data[2:end, :, n])
        if (n in local_idxs) && lock
            real_start    = 200
            real_end      = 300
            duration_drop = 30
            reduction     = 0.4 # drop covariates
            duration_rise = 30

            for v in 1:size(covs, 1)
                l        = lags[v]
                start_l  = min(real_start + l, n_times)
                end_drop = min(start_l + duration_drop, n_times)
                end_l    = min(real_end + l, n_times)

                # Gradual drop in raw space
                for t in start_l:(end_drop - 1)
                    progress   = (t - start_l + 1) / duration_drop
                    factor     = 1.0 - (1.0 - reduction) * smoothstep(progress)
                    
                    raw_val    = exp(covs[v, t]) - 1.0f0
                    covs[v, t] = log(max(0.0f0, raw_val * factor) + 1.0f0)
                end

                # Plateau at minimum for the lockdown period in raw space
                if end_drop <= end_l
                    raw_vals = exp.(covs[v, end_drop:end_l]) .- 1.0f0
                    covs[v, end_drop:end_l] .= log.(max.(0.0f0, raw_vals .* reduction) .+ 1.0f0)
                end

                # Gradual reopening in raw space
                start_rise = end_l + 1
                end_rise   = min(start_rise + duration_rise - 1, n_times)
                for t in start_rise:end_rise
                    progress   = (t - start_rise + 1) / duration_rise
                    factor     = reduction + (1.0 - reduction) * smoothstep(progress)
                    
                    raw_val    = exp(covs[v, t]) - 1.0f0
                    covs[v, t] = log(max(0.0f0, raw_val * factor) + 1.0f0)
                end
            end
        end
        for v in 1:n_var-1
            splines[v, n] = CubicSpline(tsteps, covs[v, :])
        end
    end
    return splines
end

# Prediciton function for baseline case
function predict_base(A_normal, splines_matrix, ps_model, current_st)
    n_cov = size(splines_matrix, 1)
    latents = ps_model.latent_features

    function dudt(u, p, t)
        t_c = clamp(t, 0f0, tsteps[end])
        cov_val    = Float32.([splines_matrix[v, n](t_c) for v in 1:n_cov, n in 1:n_nodes])
        u_reshaped = reshape(u, 1, n_nodes)
        input_mat  = vcat(u_reshaped, cov_val, latents)
        
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

# Prediciton function for lockdown case
function predict(A_normal, splines_matrix, ps_model, current_st, group_idxs, lock)
    n_cov = size(splines_matrix, 1)
    node_locked = [n in group_idxs for n in 1:n_nodes]
    latents = ps_model.latent_features

    function dudt(u, p, t)
        t_c = clamp(t, 0f0, tsteps[end])

        cov_val    = Float32.([splines_matrix[v, n](t_c) for v in 1:n_cov, n in 1:n_nodes])
        u_reshaped = reshape(u, 1, n_nodes)
        input_mat  = vcat(u_reshaped, cov_val, latents)
        
        if lock
            node_factors = Float32[get_isolation_factor(t_c, node_locked[n]) for n in 1:n_nodes]
            A_current = A_normal .* (node_factors * node_factors')
        else
            A_current = A_normal
        end

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

function simulate_lockdowns_model(target_groups::Vector{Vector{String}}, ps_model, current_st, A_normal; 
                                  states=all_states, X=X_norm, lags=lags, num_nodes=n_nodes, num_times=n_times)

    num_groups = length(target_groups)
    preds_group_lock = zeros(num_groups, num_times, num_nodes)

    for (i, group) in enumerate(target_groups)
        group_idxs = [findfirst(==(s), states) for s in group]
        group_idxs = filter(!isnothing, group_idxs)

        splines_lock = build_splines(X, group_idxs, lags; lock=true)
        
        pred_lock    = predict(A_normal, splines_lock, ps_model, current_st, group_idxs, true)
        preds_group_lock[i, :, :] .= pred_lock[1, :, :]
    end
    return preds_group_lock
end

function plot_scenarios(data, pred_base, pred_lock, group_idx, target_groups; denorm=false)
    group_states = target_groups[group_idx]
    name_lock    = join(group_states, "+") 

    for i in eachindex(all_states)
        state = all_states[i]
        y_label = "Daily Log Cases"

        data_curve      = data[1, :, i]
        pred_base_curve = pred_base[:, i]              
        pred_lock_curve = pred_lock[group_idx, :, i]    

        if denorm
            data_curve      = exp.(data_curve)      .- 1
            pred_base_curve = exp.(pred_base_curve) .- 1
            pred_lock_curve = exp.(pred_lock_curve) .- 1
            y_label = "Daily Cases"
        end

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

        plot!(p, tsteps, pred_base_curve, label="Baseline", lw=3, color=:blue, alpha=0.9)
        plot!(p, tsteps, pred_lock_curve, label="Lockdown: $name_lock", lw=3, color=is_locked ? :red : :green, linestyle=:solid, alpha=0.9)
        vspan!(p, [200, 300], color=:green, alpha=0.15, label="Lockdown Period")

        savefig(p, joinpath(lock_path, "gr$(group_idx)/lock_$state.png"))
        display(p)
    end
end

# -------------------------------------------------------------
# 4. Ensemble Mean Lockdowns and Predictions
# -------------------------------------------------------------

println("Loading Trained Weights from Ensemble...")
ensemble_dir = "./Parameters/Par_temporal"

ensemble_files = filter(f -> startswith(basename(f), "params_") && endswith(f, ".jld2"), readdir(ensemble_dir, join=true))
n_models = length(ensemble_files)
println("Found $n_models model parameter sets in $ensemble_dir")

ensemble_ps = []
for file in ensemble_files
    data = JLD2.load(file)
    param_key = haskey(data, "ps_final") ? "ps_final" : first(keys(data))
    push!(ensemble_ps, data[param_key])
end

target_groups = [
    ["NY", "CT", "VT", "MA", "RI", "NH", "ME"],    # Group 1: North-East
    ["CA", "WA", "OR", "NV"],                # Group 2: West Coast 
    ["TX", "FL", "LA", "MS", "AL"]                 # Group 3: South 
]

num_groups = length(target_groups)

# Samples dimension matches the number of models directly
pred_base_samples = zeros(Float32, n_models, n_times, n_nodes)
preds_lock_samples = zeros(Float32, n_models, num_groups, n_times, n_nodes)

splines_base = build_splines(X_norm, Int[], lags)  # no lockdown in baseline

latent_dim = size(ensemble_ps[1].latent_features, 1)
n_cov = n_vars - 1
nin_tot = 1 + n_cov + latent_dim
gnn = ExplicitGNN(nin_tot, 128, 1, 0.0)
_, st_gnn = Lux.setup(rng, gnn)

println("Simulating scenarios across $n_models ensemble models...")

for m_idx in 1:n_models
    ps_current = ensemble_ps[m_idx]
    
    # 1. Run baseline prediction 
    pred_base_model = predict_base(A_normal, splines_base, ps_current, st_gnn)
    pred_base_samples[m_idx, :, :] .= pred_base_model[1, :, :]

    # 2. Run lockdown simulation scenarios
    preds_lock_model = simulate_lockdowns_model(target_groups, ps_current, st_gnn, A_normal)
    preds_lock_samples[m_idx, :, :, :] .= preds_lock_model
end

# Compute Median across the ensemble dimension (dim=1)
pred_base  = dropdims(median(pred_base_samples, dims=1), dims=1)
preds_lock = dropdims(median(preds_lock_samples, dims=1), dims=1)

for i in 1:num_groups
    plot_scenarios(X_norm, pred_base, preds_lock, i, target_groups; denorm=true)
end

# -------------------------------------------------------------
# 5. Distance effect heatmap
# -------------------------------------------------------------

using PlotlyJS

pop_df = CSV.read("../Data/us_pop_by_state.csv", DataFrame)
pop_df = pop_df[:, ["state_code", "2020_census"]]
pop_df = pop_df[in.(pop_df.state_code, Ref(all_states)), :]
pop_df = pop_df[sortperm(indexin(pop_df.state_code, all_states)), :]
population = pop_df."2020_census"

t_lock_start = 200
t_lock_end   = 300
lock_range   = t_lock_start+1 : t_lock_end   

function relative_deviation(base::AbstractArray, lock::AbstractArray, trange)
    # Denormalize back to actual case counts before averaging
    b = mean(exp.(base[trange]) .- 1f0)
    l = mean(exp.(lock[trange]) .- 1f0)
    
    # Calculate relative deviation on the true scale
    return (l - b) / (abs(b) + 1f-8)   
end

rel_dev = zeros(num_groups, n_nodes)

for g in 1:num_groups
    for n in 1:n_nodes
        base_vec = pred_base[:, n]
        lock_vec = preds_lock[g, :, n]
        rel_dev[g, n] = relative_deviation(base_vec, lock_vec, lock_range)
    end
end

state_codes = all_states   
LOCKDOWN_COLOR = "firebrick"

function make_lockdown_map(group_idx::Int; rel_dev_mat=rel_dev, states=all_states, groups=target_groups, clim=(-0.10, -0.00))

    locked_set = Set(groups[group_idx])
    non_locked = [n for n in 1:length(states) if !(states[n] in locked_set)]
    
    vals = rel_dev_mat[group_idx, non_locked]

    n = length(states)
    z_vals = Vector{Float32}(undef, n)
    color_vals = Vector{String}(undef, n)
    hover_text = Vector{String}(undef, n)

    for i in 1:n
        s = states[i]
        dev = rel_dev_mat[group_idx, i] * 100   

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
    locked_z = zeros(length(locked_states))   
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

for g in 1:num_groups
    non_locked = [n for n in 1:n_nodes if !(all_states[n] in target_groups[g])]
    vals = rel_dev[g, non_locked]
    clims = [(-0.09, 0.0), (-0.24, 0.0), (-0.17, 0.0)]

    fig = make_lockdown_map(g; rel_dev_mat=rel_dev, clim = clims[g])
    display(fig)

    PlotlyJS.savefig(fig, joinpath(errors_path, "heatmap_lock_group$(g).png"),width=900, height=550)
    println("Map saved for group $g")
end


# -------------------------------------------------------------
# 6. Distance effect and graph edges correlation
# -------------------------------------------------------------

println("\n=== Computing Correlations: Lockdown Effect vs. Edge Weights ===")

for g in 1:num_groups
    locked_states = target_groups[g]
    locked_idxs = [findfirst(==(s), all_states) for s in locked_states]
    locked_idxs = filter(!isnothing, locked_idxs)
    
    non_locked_idxs = setdiff(1:n_nodes, locked_idxs)
    
    incoming_weights = Float64[]
    effects = Float64[]
    states_plotted = String[]
    
    for i in non_locked_idxs
        w = sum(A_normal[locked_idxs, i])
        push!(incoming_weights, w)
        
        e = rel_dev[g, i]
        push!(effects, e)
        
        push!(states_plotted, all_states[i])
    end
    
    pearson_corr = cor(incoming_weights, effects)
    spearman_corr = corspearman(incoming_weights, effects)
    
    group_name = join(locked_states, "+")
    println("\nGroup $g ($group_name):")
    println("  -> Pearson correlation:  $(round(pearson_corr, digits=4))")
    println("  -> Spearman correlation: $(round(spearman_corr, digits=4))")
    
    p_scatter = Plots.scatter(
        incoming_weights, effects,
        title = "Effect vs Connectivity: $group_name",
        xlabel = "Sum of Edge Weights from Locked States",
        ylabel = "Relative Deviation (Lockdown Effect)",
        label = false,
        series_annotations = text.(states_plotted, 8, :bottom),
        markercolor = :royalblue,
        markersize = 5,
        size = (800, 550),
        dpi = 150,
        left_margin = 10mm, 
        bottom_margin = 10mm
    )
    
    if length(incoming_weights) > 1
        X_mat = hcat(ones(length(incoming_weights)), incoming_weights)
        beta = X_mat \ effects
        
        x_trend = range(minimum(incoming_weights), maximum(incoming_weights), length=100)
        y_trend = beta[1] .+ beta[2] .* x_trend
        
        plot!(p_scatter, x_trend, y_trend, 
              label="Trendline (Pearson: $(round(pearson_corr, digits=2)))", 
              color=:red, lw=2)
    end
    
    display(p_scatter)
    Plots.savefig(p_scatter, joinpath(errors_path, "scatter_corr_group$(g).png"))
end

