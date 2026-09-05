# Import libraries
using Pkg
println("Current working directory: ", pwd())
Pkg.activate("..")

using CSV, DataFrames, SparseArrays, JLD2, ComponentArrays
using Graphs, Lux, GNNLux
using DifferentialEquations, DiffEqFlux
using LinearAlgebra, Statistics, Random, Plots, StatsPlots, CubicSplines
using NPZ, JSON, Dates
import Plots: plot, savefig

strt_time = time()
rng = Random.default_rng()
Random.seed!(rng, 123)

println("Running evaluation on all States with ", Threads.nthreads(), " threads")

# Create save paths
errors_path = "./Plots/errors/mlp_temporal"
pred_path = "./Plots/dynamics/mlp_temporal"
mkdir(errors_path)
mkdir(pred_path)

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
df_adj_full = CSV.read("../Data/flow_mat_exp_gravity_fitted.csv", DataFrame)
col_names = names(df_adj_full)[2:end]
n_nodes = length(all_states)
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

A = controlled_normalize(adj_sub; alpha=0.5f0)

# load data
features_raw = NPZ.npzread("../Data/data_all.npz")  
# load calendar dates
dates = JSON.parsefile("../Data/dates.json")
# Make data tensor
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
println(n_vars, n_times, n_nodes)

# Normalize
X_norm = log.(X_aligned .+ 1.0f0)
# Time
tsteps = Float32.(collect(0:n_times-1))
# Define matrix of splines functions: Num_var x Num_nodes
covariate_splines = [CubicSpline(tsteps, @view X_norm[v, :, n]) for v in 2:size(X_norm, 1), n in 1:n_nodes]

println("Data loaded. Nodes: $n_nodes, Time: $n_times, Features: $n_vars")
println("Splines ready.")

# -------------------------------------------------------------
# 2. Model Definition 
# -------------------------------------------------------------

latent_dim = 3
nin_target = 1
nin_covar = size(covariate_splines,1)
nin_tot = nin_target + nin_covar + latent_dim
nout = 1

# Frozen Dropout Layer (Unchanged)
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

# NEW LAYER (Unchanged)
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

# NEW ARCHITECTURE (Updated)
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
    # Layer 1: MLP (Dense) - Does not use adjacency matrix A
    x, st_c1 = layer.layer1(x, ps.layer1, st.layer1)
    x, st_d1 = layer.drop1(x, ps.drop1, st.drop1)

    # Layer 2: GNN - Uses adjacency matrix A
    x, st_c2 = layer.layer2(x, A, ps.layer2, st.layer2)
    x, st_d2 = layer.drop2(x, ps.drop2, st.drop2)

    # Layer 3: GNN - Uses adjacency matrix A
    x, st_c3 = layer.layer3(x, A, ps.layer3, st.layer3)
    x, st_d3 = layer.drop3(x, ps.drop3, st.drop3)

    # Layer 4: MLP (Dense) - Does not use adjacency matrix A
    x, st_c4 = layer.layer4(x, ps.layer4, st.layer4)

    new_st = (layer1=st_c1, drop1=st_d1, layer2=st_c2, drop2=st_d2, 
              layer3=st_c3, drop3=st_d3, layer4=st_c4)
    return x, new_st
end

# Initialize Model (Width=64)

println("Initializing GNN (Width 128)...")
gnn = ExplicitGNN(nin_tot, 128, nout, 0.0) 
ps_gnn, st_gnn = Lux.setup(rng, gnn)
latent_features = Lux.glorot_uniform(rng, latent_dim, n_nodes) #|>f64
ps = ComponentArray(gnn=ps_gnn, latent_features=latent_features)

# -------------------------------------------------------------
# 3. Prediciton
# -------------------------------------------------------------

split_idx = 220
split_t = tsteps[split_idx]

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

function predict_full(model, ps, st, u0, tsteps, splines)
    # Direct pass using all learned node latent features
    latents = ps.latent_features
    
    function dudt(u, p_ode, t)
        u_reshaped = reshape(u, 1, n_nodes)
        col_t = map(s -> s(t), splines)
        model_input = vcat(u_reshaped, col_t, latents)
        y, _ = model(A, model_input, p_ode.gnn, st)
        return vec(y)
    end

    prob = ODEProblem(dudt, vec(u0), (tsteps[1], tsteps[end]))
    sol = solve(prob, Tsit5(), p=ps, saveat=tsteps, reltol=1f-3, abstol=1f-3)

    sol_mat = reduce(hcat, sol.u)
    sol_reshaped = reshape(sol_mat, 1, n_nodes, length(tsteps))
    return permutedims(sol_reshaped, (1, 3, 2))
end

println("Predicting with Ensemble...")
u0 = X_norm[1, 1, :]

# Store one trajectory per ensemble model (no latent resampling needed)
pred_samples = Array{Float32}(undef, n_models, length(tsteps), n_nodes)

for m_idx in 1:n_models
    ps_current = ensemble_ps[m_idx]
    pred = predict_full(gnn, ps_current, st_gnn, u0, tsteps, covariate_splines)
    pred_samples[m_idx, :, :] .= pred[1, :, :]
end

# Summary statistics
pred_mean = dropdims(median(pred_samples, dims=1), dims=1)
pred_lower_95 = dropdims(mapslices(x -> quantile(x, 0.025), pred_samples, dims=1), dims=1)
pred_upper_95 = dropdims(mapslices(x -> quantile(x, 0.975), pred_samples, dims=1), dims=1)
pred_lower_50 = dropdims(mapslices(x -> quantile(x, 0.25), pred_samples, dims=1), dims=1)
pred_upper_50 = dropdims(mapslices(x -> quantile(x, 0.75), pred_samples, dims=1), dims=1)

# Denormalize
pred_mean = exp.(pred_mean) .- 1
pred_lower_95 = exp.(pred_lower_95) .- 1
pred_upper_95 = exp.(pred_upper_95) .- 1
pred_lower_50 = exp.(pred_lower_50) .- 1
pred_upper_50 = exp.(pred_upper_50) .- 1

# -------------------------------------------------------------
# 4. Plot predicitons
# -------------------------------------------------------------

# Ranges for segmenting plots
train_rng = 1:split_idx
test_rng = split_idx:length(tsteps) # Overlap at split_idx for continuous line drawing

for i in eachindex(all_states)
    state_name = all_states[i]
    data_curve = exp.(X_norm[1, :, i]) .- 1
    mean_curve = pred_mean[:, i]
    
    lower95, upper95 = pred_lower_95[:, i], pred_upper_95[:, i]
    lower50, upper50 = pred_lower_50[:, i], pred_upper_50[:, i]
    ribbon95 = (mean_curve .- lower95, upper95 .- mean_curve)
    ribbon50 = (mean_curve .- lower50, upper50 .- mean_curve)

    # Test Metrics (calculated strictly after time step 220)
    test_mae = mean(abs, data_curve[(split_idx+1):end] .- mean_curve[(split_idx+1):end])

    # Plot Ground Truth (Train = Black, Test = Red/Orange)
    p = plot(tsteps[train_rng], data_curve[train_rng], label="Data (Train)", color=:black, alpha=0.7, linewidth=1.5)
    plot!(p, tsteps[test_rng], data_curve[test_rng], label="Data (Test)", color=:darkorange, alpha=0.8, linewidth=1.5)

    # Plot Model Predictions & Uncertainty
    plot!(p, tsteps, mean_curve, ribbon=ribbon95, color=:blue, fillalpha=0.15, linewidth=2, label=nothing)
    plot!(p, tsteps, mean_curve, ribbon=ribbon50, color=:blue, fillalpha=0.35, linewidth=2, label="Prediction")

    # Add Vertical Line at Train/Test Cutoff
    vline!(p, [split_t], linestyle=:dash, color=:red, linewidth=1.5, label="Split (t=220)")

    title!(p, "$state_name (Test MAE: $(round(test_mae, digits=3)))")
    display(p)
    savefig(joinpath(pred_path, "dyn_state_$state_name.png"))
end


# 1. Align the dates with the lagged data
# Assuming `dates` is a Vector of strings (e.g., "2020-03-01")
parsed_dates = Date.(dates)
aligned_dates = parsed_dates[(max_lag + 1):end]
split_date = aligned_dates[split_idx]

# Ranges for segmenting plots
train_rng = 1:split_idx
test_rng = split_idx:length(aligned_dates) # Overlap at split_idx for continuous line

# 2. Select 12 states representing different geographies and population densities

selected_states = [
    "NY", "FL",  
    "IL", "PA",  
    "TX", "CA",  
    "CO", "WA",  
    "MT", "WY",  
    "MI", "OH"
]

subplots = Any[]

for state_name in selected_states
    # Find the state's index in the original all_states list
    i = findfirst(==(state_name), all_states)
    
    data_curve = exp.(X_norm[1, :, i]) .- 1
    mean_curve = pred_mean[:, i]
    
    lower95, upper95 = pred_lower_95[:, i], pred_upper_95[:, i]
    lower50, upper50 = pred_lower_50[:, i], pred_upper_50[:, i]
    ribbon95 = (mean_curve .- lower95, upper95 .- mean_curve)
    ribbon50 = (mean_curve .- lower50, upper50 .- mean_curve)

    # Test Metrics (calculated strictly after time step 220)
    test_mae = mean(abs, data_curve[(split_idx+1):end] .- mean_curve[(split_idx+1):end])

    # Plot Ground Truth with aligned_dates
    # Enabled legend, positioned it to the top left, and made the font small
    p = plot(aligned_dates[train_rng], data_curve[train_rng], 
             color=:black, alpha=0.7, linewidth=1.5, 
             label="Data (Train)", legend=:topleft, legendfontsize=6)
             
    plot!(p, aligned_dates[test_rng], data_curve[test_rng], 
          color=:darkorange, alpha=0.8, linewidth=1.5, label="Data (Test)")

    # Plot Model Predictions & Uncertainty
    plot!(p, aligned_dates, mean_curve, ribbon=ribbon95, 
          color=:blue, fillalpha=0.15, linewidth=2, label=nothing)
          
    plot!(p, aligned_dates, mean_curve, ribbon=ribbon50, 
          color=:blue, fillalpha=0.35, linewidth=2, label="Prediction")

    # Add Vertical Line at Train/Test Cutoff using the split date
    vline!(p, [split_date], linestyle=:dash, color=:red, linewidth=1.5, label="Split")

    # Add a compact title to each subplot
    title!(p, "$state_name", titlefontsize=9)
    
    # Rotate the date labels slightly for better fit
    plot!(p, xrotation=45, tickfontsize=7)

    push!(subplots, p)
end

# 3. Combine everything into a single layout
# A layout of 4 rows and 3 columns fits exactly 12 plots. 
grid_plot = plot(subplots..., layout=(4, 3), size=(1200, 800), margin=4Plots.mm)

# Display and save the grid
display(grid_plot)
savefig(grid_plot, joinpath(pred_path, "dyn_12_states_grid.png"))

# -------------------------------------------------------------
# 5. Errors tables (Temporal Split: Train t <= 220, Test t > 220)
# -------------------------------------------------------------

split_idx = 220 
train_indices = 1:split_idx 
test_indices = (split_idx + 1):n_times 
test_indices_14 = (split_idx + 1):min(split_idx + 14, n_times) 

# Load state to cluster dictionary 
state_to_cluster = load("../Data/Dict_clusters/state_cluster_mapping_expFit.jld2", "state_to_cluster") 

# Population vector for per-capita scaling 
pop_df = CSV.read("../Data/us_pop_by_state.csv", DataFrame) 
pop_df = pop_df[:, ["state_code", "2020_census"]] 
pop_df = pop_df[in.(pop_df.state_code, Ref(all_states)), :] 
pop_df = pop_df[sortperm(indexin(pop_df.state_code, all_states)), :] 
population = pop_df."2020_census" 

# Helper function for Weighted Interval Score (WIS) 
function compute_wis(y_true, lower_95, upper_95, lower_50, upper_50, median_pred) 
    function interval_score(y, l, u, alpha) 
        penalty_l = (2 / alpha) * (l - y) * (y < l) 
        penalty_u = (2 / alpha) * (y - u) * (y > u) 
        return (u - l) + penalty_l + penalty_u 
    end 

    score_95 = mean(interval_score.(y_true, lower_95, upper_95, 0.05)) 
    score_50 = mean(interval_score.(y_true, lower_50, upper_50, 0.50)) 
    ae_median = mean(abs.(y_true .- median_pred)) 

    return (0.5 * ae_median + 0.025 * score_95 + 0.25 * score_50) / (0.5 + 0.025 + 0.25) 
end 

n_samples, n_time, n_nodes = size(pred_samples) 

# 1. Sample-level evaluation matrices across ensemble runs 
# Test Horizon (t > 220) 
mae_norm_test_samples = zeros(n_samples, n_nodes) 
wmape_test_samples = zeros(n_samples, n_nodes) 
wmape_test14_samples = zeros(n_samples, n_nodes) 
per_capita_test_samples = zeros(n_samples, n_nodes) 
peak_amp_test_samples = zeros(n_samples, n_nodes) 
peak_shift_test_samples = zeros(Int, n_samples, n_nodes) 

# Train Horizon (t <= 220) Baselines 
mae_norm_train_samples  = zeros(n_samples, n_nodes) 
wmape_train_samples = zeros(n_samples, n_nodes) 

for s in 1:n_samples 
    for i in 1:n_nodes 
        # Normalized scale (training space) 
        p_norm_train = pred_samples[s, train_indices, i] 
        t_norm_train = X_norm[1, train_indices, i] 
        p_norm_test  = pred_samples[s, test_indices, i] 
        t_norm_test  = X_norm[1, test_indices, i] 
        p_norm_test14 = pred_samples[s, test_indices_14, i]
        t_norm_test14 = X_norm[1, test_indices_14, i]

        mae_norm_train_samples[s, i] = mean(abs.(p_norm_train .- t_norm_train)) 
        mae_norm_test_samples[s, i]  = mean(abs.(p_norm_test .- t_norm_test)) 

        # Denormalized scale (real case counts) 
        p_true_train = exp.(p_norm_train) .- 1.0f0 
        t_true_train = exp.(t_norm_train) .- 1.0f0 
        p_true_test  = exp.(p_norm_test) .- 1.0f0 
        t_true_test  = exp.(t_norm_test) .- 1.0f0 
        p_true_test14 = exp.(p_norm_test14) .- 1.0f0
        t_true_test14 = exp.(t_norm_test14) .- 1.0f0

        # wMAPE (Scale-Invariant Error) 
        wmape_train_samples[s, i] = sum(abs.(p_true_train .- t_true_train)) / (sum(t_true_train) + 1e-8) 
        wmape_test_samples[s, i]  = sum(abs.(p_true_test .- t_true_test)) / (sum(t_true_test) + 1e-8) 
        wmape_test14_samples[s, i] = sum(abs.(p_true_test14 .- t_true_test14)) / (sum(t_true_test14) + 1e-8)

        # Per-Capita MAE (Test horizon daily cases per 100k people) 
        mae_denorm_test = mean(abs.(p_true_test .- t_true_test)) 
        per_capita_test_samples[s, i] = (mae_denorm_test / population[i]) * 100_000 

        # Peak Dynamics (Evaluated on Test Horizon) 
        max_p = maximum(p_true_test) 
        max_t = maximum(t_true_test) 
        peak_amp_test_samples[s, i]   = (max_p - max_t) / (max_t + 1e-8) 
        peak_shift_test_samples[s, i] = argmax(p_true_test) - argmax(t_true_test) 
    end 
end 

# 2. Distributional Metrics (Evaluated across ensemble prediction bands) 
wis_test         = zeros(n_nodes) 
wis_train        = zeros(n_nodes) 
coverage_95_test = zeros(n_nodes) 
coverage_50_test = zeros(n_nodes) 

for i in 1:n_nodes 
    truth_test  = exp.(X_norm[1, test_indices, i]) .- 1.0f0 
    truth_train = exp.(X_norm[1, train_indices, i]) .- 1.0f0 

    # Empirical Coverage calibration checks (Test Horizon) 
    coverage_95_test[i] = mean((pred_lower_95[test_indices, i] .<= truth_test) .& (truth_test .<= pred_upper_95[test_indices, i])) 
    coverage_50_test[i] = mean((pred_lower_50[test_indices, i] .<= truth_test) .& (truth_test .<= pred_upper_50[test_indices, i])) 

    # Weighted Interval Scores 
    wis_test[i] = compute_wis( 
        truth_test,  
        pred_lower_95[test_indices, i], pred_upper_95[test_indices, i], 
        pred_lower_50[test_indices, i], pred_upper_50[test_indices, i], 
        pred_mean[test_indices, i] 
    ) 
    wis_train[i] = compute_wis( 
        truth_train,  
        pred_lower_95[train_indices, i], pred_upper_95[train_indices, i], 
        pred_lower_50[train_indices, i], pred_upper_50[train_indices, i], 
        pred_mean[train_indices, i] 
    ) 
end 

# 3. Construct Master DataFrame with CIs 
summary_df = DataFrame( 
    "State" => all_states, 
    "Cluster" => [get(state_to_cluster, s, "Unknown") for s in all_states], 
     
    # Calibration & Uncertainty Metrics 
    "Coverage_95_Test" => round.(coverage_95_test, digits=2), 
    "Coverage_50_Test" => round.(coverage_50_test, digits=2), 
    "WIS_Train" => round.(wis_train, digits=3), 
    "WIS_Test" => round.(wis_test, digits=3), 

    # Normalized MAE (Training Scale Space) 
    "MAE_norm_train_mean" => round.(vec(mean(mae_norm_train_samples, dims=1)), digits=4), 
    "MAE_norm_test_mean"  => round.(vec(mean(mae_norm_test_samples, dims=1)), digits=4), 
    "MAE_norm_test_q025"  => round.([quantile(col, 0.025) for col in eachcol(mae_norm_test_samples)], digits=4), 
    "MAE_norm_test_q975"  => round.([quantile(col, 0.975) for col in eachcol(mae_norm_test_samples)], digits=4), 

    # wMAPE Metrics 
    "wMAPE_train_mean" => round.(vec(mean(wmape_train_samples, dims=1)), digits=3), 
    "wMAPE_14d_mean"   => round.(vec(mean(wmape_test14_samples, dims=1)), digits=3), 
    "wMAPE_14d_q025"   => round.([quantile(col, 0.025) for col in eachcol(wmape_test14_samples)], digits=3), 
    "wMAPE_14d_q975"   => round.([quantile(col, 0.975) for col in eachcol(wmape_test14_samples)], digits=3),
    "wMAPE_test_mean"  => round.(vec(mean(wmape_test_samples, dims=1)), digits=3), 
    "wMAPE_test_q025"  => round.([quantile(col, 0.025) for col in eachcol(wmape_test_samples)], digits=3), 
    "wMAPE_test_q975"  => round.([quantile(col, 0.975) for col in eachcol(wmape_test_samples)], digits=3), 

    # Per Capita MAE (Test Horizon per 100k) 
    "PerCapita100k_test_mean" => round.(vec(mean(per_capita_test_samples, dims=1)), digits=2), 
    "PerCapita100k_test_q025" => round.([quantile(col, 0.025) for col in eachcol(per_capita_test_samples)], digits=2), 
    "PerCapita100k_test_q975" => round.([quantile(col, 0.975) for col in eachcol(per_capita_test_samples)], digits=2), 

    # Test Horizon Wave Dynamics (Peak Shift in Days & Amplitude Ratio) 
    "PeakShift_test_mean" => round.(vec(mean(peak_shift_test_samples, dims=1)), digits=1), 
    "PeakShift_test_q025" => [quantile(col, 0.025) for col in eachcol(peak_shift_test_samples)], 
    "PeakShift_test_q975" => [quantile(col, 0.975) for col in eachcol(peak_shift_test_samples)], 
     
    "PeakAmp_ratio_test_mean" => round.(vec(mean(peak_amp_test_samples, dims=1)), digits=3), 
    "PeakAmp_ratio_test_q025" => round.([quantile(col, 0.025) for col in eachcol(peak_amp_test_samples)], digits=3), 
    "PeakAmp_ratio_test_q975" => round.([quantile(col, 0.975) for col in eachcol(peak_amp_test_samples)], digits=3) 
) 

# Export Summary CSV 
CSV.write("./Tables/errors_summary_temporal.csv", summary_df) 

# 4. Aggregate Performance Printout 
println("==========================================") 
println("     TEMPORAL EVALUATION SUMMARY          ") 
println("==========================================") 
println("Test Coverage (95% CI):   ", round(mean(coverage_95_test), digits=2)) 
println("Test Coverage (50% CI):   ", round(mean(coverage_50_test), digits=2)) 
println("Mean WIS (Train / Test):  ", round(mean(wis_train), digits=3), " / ", round(mean(wis_test), digits=3)) 
println("Mean Norm MAE (Train / Test): ", round(mean(summary_df.MAE_norm_train_mean), digits=4), " / ", round(mean(summary_df.MAE_norm_test_mean), digits=4)) 
println("Mean wMAPE (Train / 14d / Test):", round(mean(summary_df.wMAPE_train_mean), digits=3), " / ", round(mean(summary_df.wMAPE_14d_mean), digits=3), " / ", round(mean(summary_df.wMAPE_test_mean), digits=3)) 
println("Mean Per-Capita MAE:      ", round(mean(summary_df.PerCapita100k_test_mean), digits=2), " cases / 100k") 
println("Mean Test Peak Shift:     ", round(mean(summary_df.PeakShift_test_mean), digits=1), " days") 
println("Mean Test Peak Amp Error: ", round(mean(summary_df.PeakAmp_ratio_test_mean) * 100, digits=1), "%")

# -------------------------------------------------------------
# 6. Errors plots
# -------------------------------------------------------------

# Boxplot comparing Training Horizon vs Testing Horizon wMAPE across all nodes
p_box = boxplot(["Train (t <= 220)" for _ in 1:n_nodes], summary_df.wMAPE_train_mean, label=false, color=:lightblue)
boxplot!(p_box, ["Test (t > 220)" for _ in 1:n_nodes], summary_df.wMAPE_test_mean, label=false, color=:coral,
         ylabel="wMAPE", title="Temporal Error Distribution (Train vs. Test)", size=(600, 400))
scatter!(p_box, repeat(["Train (t <= 220)"], n_nodes), summary_df.wMAPE_train_mean, color=:blue, alpha=0.5, label=false)
scatter!(p_box, repeat(["Test (t > 220)"], n_nodes), summary_df.wMAPE_test_mean, color=:darkred, alpha=0.5, label=false)
display(p_box)
savefig(joinpath(errors_path, "errors_train_vs_test.png"))

# Boxplot: Testing Horizon Error Distribution by Cluster
p_cluster = boxplot(summary_df.Cluster, summary_df.wMAPE_test_mean, xlabel="Cluster", ylabel="Test wMAPE",
                    title="Test Set Error Distribution by Cluster", legend=false, color=:coral)
scatter!(p_cluster, summary_df.Cluster, summary_df.wMAPE_test_mean, color=:black, alpha=0.7)
display(p_cluster)
savefig(joinpath(errors_path, "test_cluster_wmape.png"))

# Sorted Test wMAPE Plot across States
sorted_df = sort(summary_df, :wMAPE_test_mean)
p_sort = scatter(1:nrow(sorted_df), sorted_df.wMAPE_test_mean, color=:darkred, xlabel="States (sorted by Test wMAPE)",
                 ylabel="Test wMAPE", title="Test Forecast Error across States", legend=false)
xticks!(p_sort, 1:nrow(sorted_df), sorted_df.State, rotation=90)
display(p_sort)
savefig(joinpath(errors_path, "sorted_test_wmape.png"))

# Heatmap: Test errors across USA Map (PlotlyJS)
using PlotlyJS

trace_color = choropleth(
    locations = summary_df.State,
    locationmode = "USA-states",
    z = summary_df.PerCapita100k_test_mean, 
    zmin = 0,
    zmax = 200,
    colorscale = "Reds",
    marker_line_color = "white",
    colorbar_title = "Error per 100k"               # Updated colorbar label
)

trace_labels = scattergeo(
    locations = summary_df.State,
    locationmode = "USA-states",
    text = summary_df.State,
    mode = "text",
    textfont = attr(color = "black", size = 7, weight = "bold"),
    showlegend = false,
    hoverinfo = "skip"
)

layout = Layout(
    title = "Test Forecast Error per 100k across USA (t > 220)",  # Updated title
    geo = attr(scope = "usa", projection_type = "albers usa", showlakes = false),
    margin = attr(l=0, r=0, t=40, b=0)
)

p_map = PlotlyJS.plot([trace_color, trace_labels], layout, config = PlotConfig(staticPlot = true))
display(p_map)
PlotlyJS.savefig(p_map, joinpath(errors_path, "heatmap_test_per_capita.png"), width=1000, height=550)

